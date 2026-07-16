"""
Real-time two-speaker separation and toggle, using ClearVoice MossFormer2_SS_16K.
"""

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly

MODEL_SR = 16000
# Keep 0.5s windows (faster on MPS). Hop matches window so emit stays realtime.
# Seam blend does not hold back samples (holding back caused underrun glitches).
WINDOW_SEC = 0.5
HOP_SEC = 0.5
CROSSFADE_SEC = 0.06
WINDOW_SAMPLES = int(WINDOW_SEC * MODEL_SR)
HOP_SAMPLES = int(HOP_SEC * MODEL_SR)
CROSSFADE_SAMPLES = int(CROSSFADE_SEC * MODEL_SR)
BLOCK_SEC = 0.05
PLAYBACK_READY_SEC = 1.25
PLAYBACK_FLOOR_SEC = 0.0
OUTPUT_GAIN = 0.40
TRACK_TAIL_SEC = 0.4
TRACK_TAIL = int(TRACK_TAIL_SEC * MODEL_SR)
SWAP_MARGIN = 0.55
SWAP_CONFIRM = 5
QUIET_RMS = 0.018
HARD_LOCK_AFTER = 4
SWITCH_MUTE_SEC = 0.0

MODEL_NAME = "MossFormer2_SS_16K"


def load_model():
    """Load ClearVoice speech separation model; return SpeechModel wrapper."""
    from clearvoice import ClearVoice

    print(
        f"Loading ClearVoice model: {MODEL_NAME} "
        "(first run downloads pretrained weights)..."
    )
    separator = ClearVoice(task="speech_separation", model_names=[MODEL_NAME])
    speech_model = separator.models[0]
    speech_model.model.eval()
    print(f"Model loaded on {speech_model.device}.")
    dummy = torch.zeros(1, WINDOW_SAMPLES, device=speech_model.device)
    with torch.inference_mode():
        _ = speech_model.model(dummy)
        if speech_model.device.type == "mps":
            torch.mps.synchronize()
    print("Warmup done.")
    return speech_model


class RateConverter:
    """Stream-friendly sample-rate converter using resample_poly."""

    def __init__(self, from_sr, to_sr):
        self.from_sr = int(round(from_sr))
        self.to_sr = int(round(to_sr))
        if self.from_sr <= 0 or self.to_sr <= 0:
            raise ValueError(f"invalid rates: {from_sr} -> {to_sr}")
        g = math.gcd(self.from_sr, self.to_sr)
        self.up = self.to_sr // g
        self.down = self.from_sr // g
        self._buf = np.zeros(0, dtype=np.float32)

    def convert(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.from_sr == self.to_sr:
            return x
        if x.size == 0:
            return x
        self._buf = np.concatenate([self._buf, x])
        n = (len(self._buf) // self.down) * self.down
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        chunk = self._buf[:n]
        self._buf = self._buf[n:]
        y = resample_poly(chunk, self.up, self.down)
        return np.asarray(y, dtype=np.float32)


def get_target_devices():
    """Prefer built-in mic + Beats/AirPods headphones; fall back to system defaults."""
    devices = sd.query_devices()
    input_device_id = None
    output_device_id = None

    for idx, d in enumerate(devices):
        name = d["name"].lower()
        if d["max_input_channels"] > 0:
            if "built-in" in name or "macbook" in name or "imac" in name or "internal" in name:
                input_device_id = idx
                print(f"Found Mac Built-in Mic: '{d['name']}' (ID: {idx})")
                break

    if input_device_id is None:
        input_device_id = sd.default.device[0]
        print(f"No explicit built-in mic found. Using system default input (ID: {input_device_id})")

    # Prefer headphones: Beats Solo Buds, other Beats, then AirPods.
    headphone_keywords = ("beats", "solo buds", "airpods", "headphones")
    for keyword in headphone_keywords:
        for idx, d in enumerate(devices):
            name = d["name"].lower()
            if d["max_output_channels"] > 0 and keyword in name:
                output_device_id = idx
                print(f"Found headphones output: '{d['name']}' (ID: {idx})")
                break
        if output_device_id is not None:
            break

    if output_device_id is None:
        output_device_id = sd.default.device[1]
        print(f"No headphones found. Using system default output (ID: {output_device_id})")

    return input_device_id, output_device_id


def _device_sr(device_id):
    info = sd.query_devices(device_id)
    return int(round(info["default_samplerate"]))


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def _corr(x, y):
    n = min(len(x), len(y))
    if n < 16:
        return 0.0
    a = np.asarray(x[:n], dtype=np.float64)
    b = np.asarray(y[:n], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da < 1e-8 or db < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def _spectral_fingerprint(x, n_fft=1024, n_bands=48):
    """Band-averaged log-magnitude spectrum — stable speaker identity cue."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    # Average a couple of hops inside the window for stability.
    hop = n_fft // 2
    mags = []
    for start in range(0, max(1, len(x) - n_fft + 1), hop):
        frame = x[start : start + n_fft]
        if len(frame) < n_fft:
            break
        spec = np.fft.rfft(frame * np.hanning(n_fft))
        mags.append(np.abs(spec).astype(np.float64))
        if len(mags) >= 4:
            break
    if not mags:
        spec = np.fft.rfft(x[:n_fft] * np.hanning(n_fft))
        mags = [np.abs(spec).astype(np.float64)]
    mag = np.mean(mags, axis=0)
    # Collapse to coarse bands (speaker timbre, less phonetic detail).
    edges = np.linspace(0, len(mag), n_bands + 1).astype(int)
    bands = np.empty(n_bands, dtype=np.float64)
    for i in range(n_bands):
        bands[i] = np.log1p(mag[edges[i] : max(edges[i] + 1, edges[i + 1])].mean())
    bands -= bands.mean()
    bands /= np.linalg.norm(bands) + 1e-8
    return bands.astype(np.float32)


def _fp_sim(a, b):
    if a is None or b is None:
        return 0.0
    n = min(len(a), len(b))
    return float(np.dot(a[:n], b[:n]))


class SpeakerTracker:
    """Keep Speaker A/B consistent across windows (solve permutation ambiguity).

    Always routes stems by match to nearly-frozen spectral fingerprints.
    Mid-speech fingerprint updates are suppressed so identity cannot drift
    and flip. Weak / tied scores do not change the previous emit pairing
    relative to those fingerprints (large deadband while hard-locked).
    """

    def __init__(self):
        self.prev_a = None
        self.prev_b = None
        self.fp_a = None
        self.fp_b = None
        self.locked = False
        self.hard_locked = False
        self._windows = 0
        self._last_edge = 0.0

    def _scores(self, src0, src1):
        fp0 = _spectral_fingerprint(src0)
        fp1 = _spectral_fingerprint(src1)

        m0a = _fp_sim(fp0, self.fp_a)
        m0b = _fp_sim(fp0, self.fp_b)
        m1a = _fp_sim(fp1, self.fp_a)
        m1b = _fp_sim(fp1, self.fp_b)

        keep = m0a + m1b  # src0→A, src1→B
        swap = m1a + m0b  # src1→A, src0→B
        return keep, swap

    def align(self, src0, src1):
        if self.prev_a is None:
            r0, r1 = _rms(src0), _rms(src1)
            if r0 >= r1:
                stream_a, stream_b = src0, src1
            else:
                stream_a, stream_b = src1, src0
            self._update(stream_a, stream_b, update_fp=True)
            return stream_a, stream_b, False

        keep, swap = self._scores(src0, src1)
        edge = swap - keep
        r0, r1 = _rms(src0), _rms(src1)
        both_quiet = max(r0, r1) < QUIET_RMS
        both_active = min(r0, r1) > QUIET_RMS * 2.5

        # Large deadband while hard-locked + both talking: refuse weak flips.
        if self.hard_locked and both_active:
            deadband = SWAP_MARGIN * 0.75  # ~0.41 — need a clear spectral win
        elif self.hard_locked:
            deadband = SWAP_MARGIN * 0.35
        elif self.locked:
            deadband = 0.08
        else:
            deadband = 0.02

        # Sticky: if the new edge doesn't beat the deadband, reuse the sign of
        # the last *confident* edge so labels don't chatter on noise.
        if abs(edge) <= deadband:
            use_swap = self._last_edge > 0
        else:
            use_swap = edge > 0
            self._last_edge = edge

        # One speaker: pin active stem to best fingerprint (still needs margin).
        quiet_ratio = 0.1
        if r0 < quiet_ratio * max(r1, 1e-6) or r1 < quiet_ratio * max(r0, 1e-6):
            active, silent = (src1, src0) if r0 < r1 else (src0, src1)
            score_a = _fp_sim(_spectral_fingerprint(active), self.fp_a)
            score_b = _fp_sim(_spectral_fingerprint(active), self.fp_b)
            need = 0.15 if self.hard_locked else 0.06
            if score_a > score_b + need:
                routed_a, routed_b = active, silent
            elif score_b > score_a + need:
                routed_a, routed_b = silent, active
            else:
                routed_a, routed_b = (src1, src0) if use_swap else (src0, src1)
        else:
            routed_a, routed_b = (src1, src0) if use_swap else (src0, src1)

        flipped = abs(edge) > SWAP_MARGIN and (edge > 0) != (self._last_edge > 0)

        # Freeze fingerprints mid-speech — only refresh when quiet or very sure.
        update_fp = (
            not self.hard_locked
            or both_quiet
            or abs(edge) >= SWAP_MARGIN
        )
        self._update(routed_a, routed_b, update_fp=update_fp)
        return routed_a, routed_b, bool(flipped)

    def _update(self, stream_a, stream_b, update_fp=True):
        self.prev_a = stream_a.astype(np.float32, copy=True)
        self.prev_b = stream_b.astype(np.float32, copy=True)
        self._windows += 1
        self.locked = self._windows >= 2
        self.hard_locked = self._windows >= HARD_LOCK_AFTER

        if not update_fp:
            return

        fp_a = _spectral_fingerprint(stream_a)
        fp_b = _spectral_fingerprint(stream_b)
        if self.fp_a is None:
            self.fp_a, self.fp_b = fp_a, fp_b
        else:
            if self.hard_locked:
                alpha = 0.012
            elif self.locked:
                alpha = 0.05
            else:
                alpha = 0.3
            self.fp_a = (1 - alpha) * self.fp_a + alpha * fp_a
            self.fp_b = (1 - alpha) * self.fp_b + alpha * fp_b

        self.fp_a /= np.linalg.norm(self.fp_a) + 1e-8
        self.fp_b /= np.linalg.norm(self.fp_b) + 1e-8


def _emit_with_crossfade(held_tail, new_chunk, fade_len):
    """Seam-smooth abutting windows without shortening the emit (stays realtime).

    Softens the start of each new chunk toward the previous end. Emit length
    always equals the full window/hop so the playback cushion does not drain.
    """
    new_chunk = np.asarray(new_chunk, dtype=np.float32).reshape(-1).copy()
    if fade_len <= 0 or len(new_chunk) <= fade_len:
        return new_chunk, None

    if held_tail is not None:
        if len(held_tail) != fade_len:
            if len(held_tail) > fade_len:
                held_tail = held_tail[-fade_len:]
            else:
                held_tail = np.pad(held_tail, (fade_len - len(held_tail), 0))
        t = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        # Prefer new audio (0.7) so we don't smear/robotize; still kills clicks.
        fade_in = 0.35 + 0.65 * np.sin(t * (np.pi * 0.5))
        fade_out = 1.0 - fade_in
        new_chunk[:fade_len] = held_tail * fade_out + new_chunk[:fade_len] * fade_in

    return new_chunk, new_chunk[-fade_len:].copy()


def _boost_stem(x, gain=OUTPUT_GAIN, ceiling=0.65):
    """Quiet fixed gain; limit only true spikes."""
    y = np.asarray(x, dtype=np.float32) * float(gain)
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak > ceiling:
        y *= ceiling / peak
    return y.astype(np.float32)


def _separate_window(speech_model, window, mix_peak):
    """Direct model forward — skips ClearVoice's 2s padding decode path."""
    device = speech_model.device
    window_norm = (window / mix_peak).astype(np.float32)
    audio = torch.from_numpy(window_norm[None, :]).to(device)
    with torch.inference_mode():
        out_list = speech_model.model(audio)
        if device.type == "mps":
            torch.mps.synchronize()
    src0 = out_list[0][0].detach().cpu().numpy().astype(np.float32)
    src1 = out_list[1][0].detach().cpu().numpy().astype(np.float32)
    # Restore input scale only — do NOT force each source to mix RMS
    # (that was amplifying residual noise on quiet channels).
    src0 *= mix_peak
    src1 *= mix_peak
    return src0, src1


class AudioApp:
    def __init__(self):
        self.speech_model = load_model()
        self.tracker = SpeakerTracker()

        self.selected = "A"
        self.selected_lock = threading.Lock()

        self.raw_queue = queue.Queue()
        self.out_queue = queue.Queue(maxsize=400)
        self._playback_buf = np.zeros(0, dtype=np.float32)
        self._playback_lock = threading.Lock()

        self.running = False
        self.listening = False
        self.in_stream = None
        self.out_stream = None
        self.proc_thread = None
        self.playback_ready = False
        self._needs_refill = False
        self._switch_mute_until = 0.0

        self.input_sr = MODEL_SR
        self.output_sr = MODEL_SR
        self.in_resampler = RateConverter(MODEL_SR, MODEL_SR)
        self.out_resampler = RateConverter(MODEL_SR, MODEL_SR)

        self._model_sr_buffer = np.zeros(0, dtype=np.float32)
        self._fade_tail_a = None
        self._fade_tail_b = None

        self.stats_lock = threading.Lock()
        self.stats = {
            "in_rms": 0.0,
            "out_rms": 0.0,
            "infer_s": 0.0,
            "windows": 0,
            "swaps": 0,
            "buf_s": 0.0,
            "paused": 0,
        }

    def input_callback(self, indata, frames, time_info, status):
        if status:
            print("input status:", status)
        mono = indata[:, 0].copy()
        self.raw_queue.put(mono)

    def _drain_playback(self):
        """Clear queued/output audio so a switch can rebuffer cleanly."""
        with self._playback_lock:
            self._playback_buf = np.zeros(0, dtype=np.float32)
            self.playback_ready = False
        while not self.out_queue.empty():
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                break

    def output_callback(self, outdata, frames, time_info, status):
        if status:
            print("output status:", status)

        ready_need = int(self.output_sr * PLAYBACK_READY_SEC)

        with self._playback_lock:
            while True:
                try:
                    nxt = self.out_queue.get_nowait()
                    self._playback_buf = np.concatenate([self._playback_buf, nxt])
                except queue.Empty:
                    break

            buf_len = len(self._playback_buf)

            if not self.playback_ready:
                if buf_len >= ready_need:
                    self.playback_ready = True
                else:
                    outdata.fill(0.0)
                    return

            # Always emit a full block. If short, hold last sample then fade —
            # hard zero pads were the intermittent robotic glitches.
            chunk = np.zeros(frames, dtype=np.float32)
            take = min(frames, len(self._playback_buf))
            if take:
                chunk[:take] = self._playback_buf[:take]
                self._playback_buf = self._playback_buf[take:]
                self._last_out_sample = float(chunk[take - 1])
            if take < frames:
                # Soft hold of last sample decaying to zero (no digital zip).
                hold = getattr(self, "_last_out_sample", 0.0)
                remain = frames - take
                decay = np.linspace(1.0, 0.0, remain, dtype=np.float32)
                chunk[take:] = hold * decay * 0.15
                self._last_out_sample = 0.0
                with self.stats_lock:
                    self.stats["paused"] += 1

            with self.stats_lock:
                self.stats["buf_s"] = len(self._playback_buf) / max(self.output_sr, 1)

        outdata[:, 0] = chunk
        if outdata.shape[1] > 1:
            outdata[:, 1] = chunk

    def processing_loop(self):
        while self.running:
            try:
                raw = self.raw_queue.get(timeout=1)
            except queue.Empty:
                continue

            model_audio = self.in_resampler.convert(raw)
            if model_audio.size == 0:
                continue

            with self.stats_lock:
                self.stats["in_rms"] = _rms(model_audio)

            self._model_sr_buffer = np.concatenate([self._model_sr_buffer, model_audio])

            # If inference fell behind, drop old mic audio so we stay near-live
            # instead of processing a long backlog of stale windows in a burst.
            max_buf = WINDOW_SAMPLES + HOP_SAMPLES * 2
            if len(self._model_sr_buffer) > max_buf:
                self._model_sr_buffer = self._model_sr_buffer[-max_buf:]

            # Only run another separation if playback cushion needs audio.
            with self._playback_lock:
                buffered = len(self._playback_buf)
            queued = self.out_queue.qsize()
            # Rough output seconds already waiting.
            out_wait = buffered / max(self.output_sr, 1) + queued * HOP_SEC
            if out_wait > PLAYBACK_READY_SEC * 1.4 and len(self._model_sr_buffer) >= WINDOW_SAMPLES:
                # Healthy buffer — trim mic backlog and wait.
                if len(self._model_sr_buffer) > WINDOW_SAMPLES:
                    self._model_sr_buffer = self._model_sr_buffer[-WINDOW_SAMPLES:]
                continue

            while len(self._model_sr_buffer) >= WINDOW_SAMPLES:
                window = self._model_sr_buffer[:WINDOW_SAMPLES]
                self._model_sr_buffer = self._model_sr_buffer[HOP_SAMPLES:]

                mix_peak = float(np.max(np.abs(window)) + 1e-8)
                # Avoid insane gain on near-silence (noise → robotic bursts).
                mix_peak = max(mix_peak, 0.02)

                t0 = time.perf_counter()
                src0, src1 = _separate_window(self.speech_model, window, mix_peak)
                infer_s = time.perf_counter() - t0
                if infer_s > HOP_SEC * 1.05:
                    print(
                        f"note: inference {infer_s:.2f}s > hop {HOP_SEC:.2f}s "
                        "(dropping backlog rather than glitching)"
                    )

                n = min(len(src0), len(src1), WINDOW_SAMPLES)
                src0 = src0[:n]
                src1 = src1[:n]
                if n < WINDOW_SAMPLES:
                    pad = WINDOW_SAMPLES - n
                    src0 = np.pad(src0, (0, pad))
                    src1 = np.pad(src1, (0, pad))

                stream_a, stream_b, swapped = self.tracker.align(src0, src1)
                stream_a = _boost_stem(stream_a)
                stream_b = _boost_stem(stream_b)

                new_a, self._fade_tail_a = _emit_with_crossfade(
                    self._fade_tail_a, stream_a, CROSSFADE_SAMPLES
                )
                new_b, self._fade_tail_b = _emit_with_crossfade(
                    self._fade_tail_b, stream_b, CROSSFADE_SAMPLES
                )

                with self.selected_lock:
                    chosen = new_a if self.selected == "A" else new_b

                out_chunk = self.out_resampler.convert(chosen.astype(np.float32))
                with self.stats_lock:
                    self.stats["infer_s"] = infer_s
                    self.stats["windows"] += 1
                    if swapped:
                        self.stats["swaps"] += 1
                    self.stats["out_rms"] = _rms(out_chunk) if out_chunk.size else 0.0

                if out_chunk.size == 0:
                    continue

                try:
                    self.out_queue.put(out_chunk, timeout=2.0)
                except queue.Full:
                    print("note: output queue full — skipping window")

                # One window per mic read keeps pacing stable; loop again only
                # if the cushion is critically low.
                with self._playback_lock:
                    buffered = len(self._playback_buf)
                if buffered / max(self.output_sr, 1) + self.out_queue.qsize() * HOP_SEC > 0.35:
                    break

    def set_selected(self, choice):
        with self.selected_lock:
            if choice == self.selected:
                return
            self.selected = choice
        # Keep playing — hard drain/mute was causing toggle clicks.
        # Fade tails stay so the next window still seams cleanly.

    def _open_streams(self, input_device, output_device):
        self.input_sr = _device_sr(input_device)
        self.output_sr = _device_sr(output_device)
        self.in_resampler = RateConverter(self.input_sr, MODEL_SR)
        self.out_resampler = RateConverter(MODEL_SR, self.output_sr)

        print(
            f"Audio: input #{input_device} @ {self.input_sr} Hz → "
            f"model {MODEL_SR} Hz → output #{output_device} @ {self.output_sr} Hz"
        )

        in_blocksize = max(64, int(self.input_sr * BLOCK_SEC))
        out_blocksize = max(64, int(self.output_sr * BLOCK_SEC))

        out_info = sd.query_devices(device=output_device)
        out_channels = 2 if out_info["max_output_channels"] >= 2 else 1

        in_stream = sd.InputStream(
            device=input_device,
            samplerate=self.input_sr,
            channels=1,
            dtype="float32",
            callback=self.input_callback,
            blocksize=in_blocksize,
        )
        out_stream = sd.OutputStream(
            device=output_device,
            samplerate=self.output_sr,
            channels=out_channels,
            dtype="float32",
            callback=self.output_callback,
            blocksize=out_blocksize,
        )
        return in_stream, out_stream

    def start_streams(self):
        if self.listening:
            return

        self.playback_ready = False
        self._needs_refill = False
        self._switch_mute_until = 0.0
        self.tracker = SpeakerTracker()
        self._model_sr_buffer = np.zeros(0, dtype=np.float32)
        self._fade_tail_a = None
        self._fade_tail_b = None

        self._drain_playback()
        while not self.raw_queue.empty():
            try:
                self.raw_queue.get_nowait()
            except queue.Empty:
                break

        preferred_in, preferred_out = get_target_devices()
        candidates = [
            (preferred_in, preferred_out),
            (preferred_in, sd.default.device[1]),
            (sd.default.device[0], sd.default.device[1]),
        ]

        last_err = None
        in_stream = None
        out_stream = None
        for input_device, output_device in candidates:
            try:
                in_stream, out_stream = self._open_streams(input_device, output_device)
                in_stream.start()
                out_stream.start()
                self.in_stream = in_stream
                self.out_stream = out_stream
                break
            except Exception as err:
                last_err = err
                print(f"Failed to open devices in={input_device} out={output_device}: {err}")
                try:
                    if in_stream is not None:
                        in_stream.close()
                except Exception:
                    pass
                try:
                    if out_stream is not None:
                        out_stream.close()
                except Exception:
                    pass
                in_stream = None
                out_stream = None
        else:
            raise RuntimeError(f"Could not open audio streams: {last_err}") from last_err

        self.running = True
        self.listening = True
        self.proc_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.proc_thread.start()

    def stop(self):
        self.running = False
        self.listening = False
        try:
            if self.in_stream is not None:
                self.in_stream.stop()
                self.in_stream.close()
            if self.out_stream is not None:
                self.out_stream.stop()
                self.out_stream.close()
        except Exception:
            pass
        self.in_stream = None
        self.out_stream = None


def build_gui(app: AudioApp):
    root = tk.Tk()
    root.title("Wavelength - Dual Speaker Control")
    root.geometry("460x340")

    status = tk.StringVar(value="Idle — press Start Listening")
    meters = tk.StringVar(value="mic: —   out: —   infer: —")

    def choose(letter):
        if not app.listening:
            return
        app.set_selected(letter)
        status.set(f"Listening to: Speaker {letter}")

    label = ttk.Label(root, textvariable=status, font=("Helvetica", 14))
    label.pack(pady=10)

    meter_label = ttk.Label(root, textvariable=meters, font=("Helvetica", 11), foreground="gray")
    meter_label.pack(pady=4)

    def poll_stats():
        with app.stats_lock:
            s = dict(app.stats)
        if app.listening:
            meters.set(
                f"mic: {s['in_rms']:.3f}   out: {s['out_rms']:.3f}   "
                f"infer: {s['infer_s']:.2f}s   buf: {s['buf_s']:.2f}s   "
                f"flips~{s['swaps']}   pause~{s['paused']}"
            )
        root.after(200, poll_stats)

    def start_listening():
        start_btn.config(state=tk.DISABLED)
        try:
            app.start_streams()
        except Exception as err:
            start_btn.config(state=tk.NORMAL)
            status.set("Failed to start audio")
            messagebox.showerror("Audio Error", str(err))
            return
        btn_a.config(state=tk.NORMAL)
        btn_b.config(state=tk.NORMAL)
        status.set(f"Listening to: Speaker {app.selected}")

    start_btn = tk.Button(
        root,
        text="Start Listening",
        font=("Helvetica", 14),
        width=20,
        height=2,
        command=start_listening,
    )
    start_btn.pack(pady=6)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)

    btn_a = tk.Button(
        btn_frame,
        text="Speaker A",
        font=("Helvetica", 14),
        width=14,
        height=2,
        command=lambda: choose("A"),
        state=tk.DISABLED,
    )
    btn_a.grid(row=0, column=0, padx=10)

    btn_b = tk.Button(
        btn_frame,
        text="Speaker B",
        font=("Helvetica", 14),
        width=14,
        height=2,
        command=lambda: choose("B"),
        state=tk.DISABLED,
    )
    btn_b.grid(row=0, column=1, padx=10)

    note = ttk.Label(
        root,
        text="True A vs B separation (MossFormer2).\n"
        "Identity lock resists mid-sentence flips.",
        justify="center",
        foreground="gray",
    )
    note.pack(pady=10)

    def on_close():
        app.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(200, poll_stats)
    return root


def main():
    app = AudioApp()
    root = build_gui(app)
    root.mainloop()


if __name__ == "__main__":
    main()
