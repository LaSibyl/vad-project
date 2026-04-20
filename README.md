# Voice Activity Detection (VAD) Baseline

## Overview

This project implements and evaluates two VAD approaches on real audio:

1. **Energy-based VAD** — log frame energy + threshold + median smoothing (baseline)
2. **CNN-based VAD** — lightweight 1-D CNN ($<$500 K parameters) trained on LibriSpeech + MUSAN, evaluated on real recordings

Final performance on `data/sample.wav`:

| Method | F1 | FAR | MISS |
|---|---|---|---|
| Energy VAD (raw) | 0.899 | 0.039 | 0.090 |
| Energy VAD (smoothed) | 0.922 | 0.022 | 0.090 |
| CNN VAD (threshold = 0.5) | **0.932** | 0.002 | 0.122 |
| CNN VAD (threshold = 0.1) | **0.945** | 0.015 | 0.064 |

---

## Repository Structure

```
vad_baseline_demo/
├── src/
│   ├── energy_vad_demo.py        # Energy VAD pipeline (framing, threshold, smoothing)
│   ├── vad_dl_demo.py            # CNN model definition (LightweightCNN_VAD) + synthetic dataset
│   ├── build_real_dataset.py     # Dataset builder: LibriSpeech + MUSAN + real silence extraction
│   ├── train_on_real_data.py     # Training script (BCEWithLogitsLoss, z-score normalisation)
│   ├── dl_inference.py           # CNN inference on real audio (numpy feature extractor)
│   ├── compare_models.py         # Unified evaluation: Energy vs CNN on sample.wav
│   ├── evaluation.py             # Shared metrics (F1, FAR, MISS)
│   └── diagnostics.py            # Feature distribution diagnostics
├── checkpoints/
│   ├── cnn_real_data.pt          # Trained CNN checkpoint
│   └── feature_scaler.pt         # Per-feature z-score scaler (mean/std from training set)
├── data/
│   ├── raw/                      # LibriSpeech (train-clean-5, dev-clean-2), MUSAN, RIRS_NOISES
│   └── processed/                # Cached feature tensors (train_X.pt, train_y.pt, val_X.pt, val_y.pt)
├── outputs/                      # Generated plots (probability distribution, feature spaces)
├── reports/                      # LaTeX reports and PDFs
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Raw data must be placed under `data/raw/` before building the dataset:
- `LibriSpeech/train-clean-5/` and `LibriSpeech/dev-clean-2/`
- `musan/noise/`

---

## Usage

### 1. Build dataset
```bash
python src/build_real_dataset.py
```
Extracts features from LibriSpeech + MUSAN and caches them to `data/processed/`.
Non-speech samples include MUSAN noise **and** real low-energy silence segments
mined from the LibriSpeech recordings (RMS < 0.005).

### 2. Train CNN
```bash
python src/train_on_real_data.py
```
Trains `LightweightCNN_VAD` for 20 epochs using `BCEWithLogitsLoss`.
Saves model to `checkpoints/cnn_real_data.pt` and scaler to `checkpoints/feature_scaler.pt`.

### 3. Evaluate
```bash
python src/compare_models.py
```
Runs Energy VAD and CNN VAD on `data/sample.wav`, prints F1/FAR/MISS for all methods,
performs threshold sweep, and saves a probability distribution plot to `outputs/`.

---

## Key Design Decisions

- **No Sigmoid in model** — `LightweightCNN_VAD` outputs raw logits; `BCEWithLogitsLoss`
  is used during training for numerical stability. `torch.sigmoid()` is applied at inference.
- **Unified feature extractor** — both training (`build_real_dataset.py`) and inference
  (`dl_inference.py`) use the same numpy FFT pipeline (Hann window, 512-point FFT, 40 mel bins,
  5-frame context window → 200-dim feature vector).
- **Real silence in training** — non-speech class includes genuine recording pause segments
  to prevent the model from classifying real silence as speech (key fix for FAR=0.96 → 0.002).
- **Z-score normalisation** — a global per-feature scaler is fit on the training split
  and saved alongside the checkpoint for consistent inference-time normalisation.

---

## Results History

| Commit | Fix | Real-audio F1 |
|---|---|---|
| `30e5abe` | Baseline (real data training, BCELoss+sigmoid) | 0.050 |
| `77591ee` | BCEWithLogitsLoss, remove double sigmoid | val 0.92 / real 0.050 |
| `fd4e806` | Unified numpy feature extractor | 0.42 |
| `f4023db` | Real silence samples in training data | **0.932** |
