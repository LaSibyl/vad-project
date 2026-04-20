# Voice Activity Detection — Energy VAD vs CNN VAD

## Research Question

> Under real-world noise conditions, does a lightweight CNN-based VAD offer
> meaningfully better stability than a classical energy-based VAD, and at what cost?

This project implements, trains, debugs, and systematically compares two VAD approaches
across varying signal-to-noise ratios (SNR):

1. **Energy VAD** — log frame energy + adaptive threshold + median filter smoothing. Zero training cost, 0.005 ms/frame.
2. **CNN VAD** — lightweight 1-D CNN (<500 K params) trained on LibriSpeech + MUSAN noise, with log-mel spectrogram features. 0.34 ms/frame.

---

## Experimental Design

### Comparison axes
- **Peak accuracy** — F1 on clean audio (`data/sample.wav`)
- **SNR stability** — F1 and FAR swept over SNR ∈ {0, 5, 10, 15, 20} dB (MUSAN noise mixing)
- **Worst-case robustness** — minimum F1 across all SNR levels
- **Inference latency** — per-frame CPU time

### Evaluation audio
`data/sample.wav` (6.21 s, 16 kHz) with ground truth speech region 2.6 s – 4.15 s.
For SNR sweep: speech signal is mixed with MUSAN noise at controlled SNR levels;
ground truth labels are held fixed.

### Training data (CNN only)
Four sample categories (25% each per batch):
| Category | Source |
|---|---|
| Speech only | LibriSpeech train-clean-5 |
| Noise only | MUSAN noise |
| Speech + noise mix | LibriSpeech + MUSAN (SNR 5–20 dB) |
| Real silence | Low-energy LibriSpeech pause segments (RMS < 0.005) |

---

## Results

### Clean audio (single recording)

| Method | F1 | FAR | MISS |
|---|---|---|---|
| Energy VAD (raw) | 0.899 | 0.039 | 0.090 |
| Energy VAD (smoothed) | 0.922 | 0.022 | 0.090 |
| CNN VAD (threshold = 0.5) | **0.932** | 0.002 | 0.122 |
| CNN VAD (threshold = 0.1) | **0.945** | 0.015 | 0.064 |

### SNR stability sweep (0 – 20 dB)

| SNR (dB) | Energy F1 | Energy FAR | CNN F1 | CNN FAR |
|---|---|---|---|---|
| 0 | 0.612 | 0.320 | 0.569 | 0.503 |
| 5 | 0.621 | 0.315 | 0.644 | 0.319 |
| 10 | 0.638 | 0.309 | 0.670 | 0.259 |
| 15 | 0.647 | 0.296 | 0.831 | 0.075 |
| 20 | 0.683 | 0.251 | **0.922** | 0.007 |

| Metric | Energy mean ± std | CNN mean ± std |
|---|---|---|
| F1 | 0.640 ± 0.025 | 0.727 ± 0.130 |
| FAR | 0.298 ± 0.025 | 0.233 ± 0.177 |
| Latency | **0.005 ms/frame** | 0.340 ms/frame (71× slower) |

### Key finding
Energy VAD is **stable but capped** (F1 std = 0.025, max F1 = 0.683).
CNN VAD is **high-ceiling but sensitive** (F1 std = 0.130, max F1 = 0.922 at 20 dB, worst-case 0.569 at 0 dB).
The crossover point is ~10 dB SNR — below that, Energy VAD is more reliable.

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
│   ├── compare_models.py         # Single-recording evaluation: Energy vs CNN on sample.wav
│   ├── snr_stability_eval.py     # SNR sweep: F1/FAR vs SNR + latency measurement
│   ├── evaluation.py             # Shared metrics (F1, FAR, MISS)
│   └── diagnostics.py            # Feature distribution diagnostics
├── checkpoints/
│   ├── cnn_real_data.pt          # Trained CNN checkpoint
│   └── feature_scaler.pt         # Per-feature z-score scaler (mean/std from training set)
├── data/
│   ├── raw/                      # LibriSpeech (train-clean-5, dev-clean-2), MUSAN, RIRS_NOISES
│   └── processed/                # Cached feature tensors (train_X.pt, train_y.pt, val_X.pt, val_y.pt)
├── outputs/
│   ├── snr_stability.png         # F1 and FAR vs SNR curves (main stability result)
│   └── probability_distribution.png
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

### 3. Single-recording evaluation
```bash
python src/compare_models.py
```
Runs Energy VAD and CNN VAD on `data/sample.wav`, prints F1/FAR/MISS for all methods,
performs threshold sweep, and saves a probability distribution plot to `outputs/`.

### 4. SNR stability sweep
```bash
python src/snr_stability_eval.py
```
Mixes `data/sample.wav` with MUSAN noise at SNR ∈ {0, 5, 10, 15, 20} dB, runs both VADs
at each level, prints F1/FAR/MISS tables + frame-level latency, and saves
`outputs/snr_stability.png`.

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

## Practical Trade-off

| Scenario | Recommended |
|---|---|
| SNR < 10 dB (very noisy) | Energy VAD — more stable, no worst-case collapse |
| SNR ≥ 10 dB (office / phone) | CNN VAD — F1 0.831–0.922 vs Energy 0.638–0.683 |
| Embedded / no PyTorch | Energy VAD — numpy only, 71× lower latency |
| Strict FAR requirement | CNN VAD — FAR 0.007 vs Energy 0.251 at 20 dB |
| No labelled training data | Energy VAD — zero training cost |

**Cascade design (future work):** Energy VAD pre-screens frames (fast, stable); CNN runs only
on frames that pass the energy gate — combines low latency with high-SNR precision.

---

## CNN Training History

| Commit | Fix | Real-audio F1 |
|---|---|---|
| `30e5abe` | Baseline (real data training, BCELoss+sigmoid) | 0.050 |
| `77591ee` | BCEWithLogitsLoss, remove double sigmoid | val 0.92 / real 0.050 |
| `fd4e806` | Unified numpy feature extractor | 0.42 |
| `f4023db` | Real silence samples in training data | **0.932** |
