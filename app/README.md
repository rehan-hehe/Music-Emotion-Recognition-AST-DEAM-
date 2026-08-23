# Sonic Compass — Music Emotion Recognition

A Streamlit app that deploys the AST (Audio Spectrogram Transformer) model
trained on DEAM to predict a song's **valence** and **arousal** (1–9 scale)
and plot it on the emotion circumplex.

## Files

- **`predict.py`** — model loading and audio preprocessing. This is the
  verified-working backend: `librosa.load(..., sr=44100, mono=True)` for
  decoding/resampling/mono-mixing in one step, and `urllib.request.urlretrieve`
  for fetching the weights. If you need to run inference from the command
  line, `python predict.py path/to/song.mp3` still works standalone.
- **`app.py`** — the Streamlit UI. Imports `load_model` and `predict_emotion`
  from `predict.py` directly rather than re-implementing them, so the app
  and the CLI script can never drift out of sync on preprocessing.
- **`requirements.txt`** / **`packages.txt`** — Python and system dependencies.

## What changed from the previous version

- **Audio loading now goes through `librosa`** instead of `torchaudio.load`
  + manual resampling. `librosa.load(path, sr=44100, mono=True)` handles
  decoding, resampling, and mono-mixing in one call and is more robust
  across MP3/WAV/FLAC/M4A than the previous `torchaudio` backend, which is
  almost certainly what was causing the model/audio loading failures.
- **Weight download uses `urllib.request.urlretrieve`** rather than a
  streamed `requests` call, matching the version confirmed to work.
- **Normalization constants updated** to the exact values from `predict.py`:
  `GLOBAL_MEAN = -33.7032`, `GLOBAL_STD = 37.3519` (previously
  `-33.6747` / `37.4105` — these were pulled from the notebook's printed
  output, but the values in `predict.py` are stated as the exact ones used
  for the deployed checkpoint, so those are the ones now in use).
- **Short clips now fail with a clear message instead of being looped.**
  `predict.py` raises `ValueError` if the spectrogram has fewer than 9,216
  frames (~41.8 seconds) — matching how the model was actually validated —
  and the app catches this and shows a friendly "clip too short" message
  rather than fabricating audio by looping.
- **Display scale reverted to the original DEAM 1–9 scale** (`raw * 8 + 1`),
  matching the conversion already used in `predict.py`'s own CLI output,
  rather than an arbitrary 0–10 rescaling.
- One small **additive** change to `predict.py`: `predict_emotion` now
  returns the raw per-chunk predictions (`[9, 2]` array) in addition to the
  averaged valence/arousal, so the app can plot how the predicted emotion
  moves across the song. This doesn't change any preprocessing, model
  logic, or the final averaged result — `main()` was updated to just
  ignore the extra return value, so the CLI output is identical to before.

## Preprocessing, exactly as trained

1. `librosa.load(path, sr=44100, mono=True)`
2. `torchaudio.transforms.MelSpectrogram(sample_rate=44100)` → `AmplitudeToDB()`
3. Reject clips under 9,216 frames (~41.8s)
4. Truncate to the first 9,830 frames
5. Normalize: `(spec - (-33.7032)) / 37.3519`
6. Truncate to 9,216 frames, split into 9 chunks of 1,024 frames (~4.64s each)
7. Each chunk scored independently, `[9, 1, 128, 1024]` → model → `[9, 2]`
8. Averaged across chunks, then `raw * 8 + 1` back to the 1–9 DEAM scale

## Model performance (why AST was chosen)

From the held-out test comparison (2,430 chunk-level test examples):

| Model | Valence MAE (1-9) | Arousal MAE (1-9) | Valence CCC | Arousal CCC |
|---|---|---|---|---|
| **AST (Transformer)** | 0.7738 | 0.7340 | **0.5986** | **0.6756** |
| LSTM | 0.7579 | 0.7325 | 0.5590 | 0.6389 |
| CNN | 0.7959 | 0.7915 | 0.4814 | 0.5684 |
| Dummy baseline | 0.9843 | 1.0551 | 0.0000 | 0.0000 |
| MLP | 0.9863 | 1.0546 | 0.0000 | 0.0006 |

AST has the best CCC (Concordance Correlation Coefficient) on both targets,
even though the LSTM has a marginally lower raw MAE. The app's sidebar shows
this full table rather than hiding the nuance.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model weights (~350MB) and caches them via
`st.cache_resource`, so subsequent runs start much faster.

## Deploying

**Streamlit Community Cloud** — push this folder to a GitHub repo and point
a new app at `app.py`. `requirements.txt` and `packages.txt` are picked up
automatically.

**Hugging Face Spaces** — create a Streamlit-SDK Space and upload `app.py`,
`predict.py`, `requirements.txt`, and `packages.txt` (as `apt.txt` if the
Space requires that name).
