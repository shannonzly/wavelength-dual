# Real-Time Two-Speaker Toggle with ClearVoice

Listen to your microphone with two overlapping voices, and toggle which
speaker you hear, live — powered by
[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
(`MossFormer2_SS_16K`).

## About the model

This app uses ClearVoice’s pretrained speech-separation model:

- Model: `MossFormer2_SS_16K`
- Task: `speech_separation` (fixed at 2 speakers, 16 kHz)
- Install: `pip install clearvoice` — weights download automatically on first run
  from HuggingFace into ClearVoice’s checkpoint cache

## Requirements

- Python 3.9+
- A working microphone and speakers/headphones (**use headphones** — without
  them, the speaker output can feed back into the mic)
- On Linux, tkinter may need a separate package:
  `sudo apt install python3-tk`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python realtime_voice_separator.py
```

The first run downloads the pretrained model. After that it loads from cache.

## Using it

1. A small window opens with two buttons: "Speaker A" and "Speaker B".
2. Have two people speak into the mic (ideally close together, e.g. both
   near a laptop mic, or a single mic picking up both).
3. Click a button to route that person's separated voice to your speakers.
   You can click back and forth freely while both people keep talking.

## Important limitations

- **Latency**: ~1–2 seconds. MossFormer2 is not a causal/streaming model —
  it needs a short window of audio to separate speakers, so there is an
  inherent delay.
- **Speakers may swap labels**: the network re-solves separation on each
  ~1s window. This app aligns consecutive chunks by correlating the overlap
  so A/B usually stay consistent, but brief silence can flip labels. If
  that happens, click the other button.
- **Quality depends on your mic setup**: clean, close two-speaker speech at
  16 kHz works best; heavy noise, music, or strong reverb hurts results.
- **Two speakers only**: `MossFormer2_SS_16K` is fixed at 2 sources.
- **Performance**: MossFormer2 is heavier than smaller separators. On Apple
  Silicon, PyTorch MPS may help when available; otherwise CPU can fall
  behind the hop interval and you may hear dropouts.

## Files

- `realtime_voice_separator.py` — the app (run this)
- `requirements.txt` — pip dependencies
