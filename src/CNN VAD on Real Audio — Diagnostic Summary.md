# CNN VAD on Real Audio — Diagnostic Summary

## 1. Objective

This experiment aims to evaluate the performance of a CNN-based Voice Activity Detection (VAD) model on real audio data and compare it with a traditional Energy-based VAD baseline.

---

## 2. Experimental Setup

### Models
- **Energy VAD**
  - Raw version
  - Smoothed version

- **CNN VAD**
  - Trained on synthetic data
  - Lightweight CNN (< 500K parameters)
  - Input: Log-Mel features (40 mel bins × 5 frames)

### Evaluation Data
- Energy VAD: real `sample.wav` with manual ground truth
- CNN VAD:
  - Synthetic validation (during training)
  - Real audio (via unified pipeline)

### Alignment Fix (Important)
To ensure fair comparison:
- Used `librosa.util.frame(hop_length=160)` for consistent framing
- Applied context window alignment (`gt[2:-2]`)
- Verified:

GT (aligned): 615
DL preds : 615
✓ Lengths match


---

## 3. Results

### Energy VAD

| Method             | F1     | FAR     | MISS    |
|------------------|--------|--------|--------|
| Energy (Raw)      | 0.8987 | 0.0389 | 0.0897 |
| Energy (Smoothed) | 0.9221 | 0.0216 | 0.0897 |

---

### CNN VAD (Synthetic)

| Dataset   | F1     | FAR   | MISS |
|----------|--------|-------|------|
| Synthetic | 1.0000 | 0.000 | 0.000 |

✅ Perfect performance on synthetic validation

---

### CNN VAD (Real Audio)

| Metric | Value |
|--------|------|
| F1     | 0.4047 |
| FAR    | 1.0000 ⚠️ |
| MISS   | 0.0000 |

Confusion matrix:

TN=0, FP=459, FN=0, TP=156


---

## 4. Key Observation

The CNN model predicts **all frames as speech (positive class)**:

- 100% False Alarm Rate
- 0% Miss Rate
- No true negatives

---

## 5. Diagnostic Analysis

### 5.1 Threshold Sweep

Tested thresholds from 0.1 to 0.9:

| Threshold | F1     | FAR    | MISS   |
|----------|--------|--------|--------|
| 0.1–0.9  | 0.4047 | 1.0000 | 0.0000 |

📌 **Observation:**
- Performance is invariant to threshold
- Indicates threshold is NOT the root cause

---

### 5.2 Probability Distribution

Statistics:


Min probability : 1.0000
Max probability : 1.0000
Mean probability : 1.0000
Std deviation : 0.0000


📌 **Observation:**
- All outputs are exactly 1.0
- Model is fully saturated
- No discrimination capability on real data

---

### 5.3 Visualization Insight

- Time-series plot: flat line at probability = 1.0
- Histogram: all mass concentrated at 1.0

📌 Confirms:
> The model collapses into an **all-positive classifier** on real audio.

---

## 6. Interpretation

This failure is **NOT due to threshold selection**, but rather:

### Root Cause Hypothesis

#### (1) Severe domain shift
- Model trained on synthetic data
- Real audio distribution significantly different

#### (2) Feature mismatch
- Log-Mel features from real audio likely differ in scale/distribution
- Model input during inference not aligned with training distribution

#### (3) Overfitting to synthetic patterns
- Synthetic task too simple (F1 = 1.0 across all strategies)
- Model learns shortcuts instead of generalizable speech patterns

---

## 7. Conclusion

- CNN VAD performs perfectly on synthetic data but **fails completely on real audio**
- The model outputs are:
  > **Fully saturated (all 1.0), resulting in 100% false alarm rate**

- Threshold tuning is ineffective
- The issue lies in:
  > **feature distribution mismatch and lack of generalization**

---

## 8. Next Steps

### Immediate (High Priority)
- Compare feature distributions:
  - Synthetic vs. real Log-Mel inputs
- Normalize / standardize input features consistently

### Short-term
- Introduce adaptive thresholding or calibration (optional)
- Debug `run_dl_on_audio()` input pipeline

### Medium-term
- Mix synthetic + real data in training
- Add realistic noise augmentation

### Long-term
- Train on standard datasets (e.g., LibriSpeech + MUSAN)
- Build unified evaluation framework across methods

---

## 9. Summary (One Sentence)

> The CNN model fails on real audio because its outputs collapse to a constant value (1.0), revealing a fundamental mismatch between training (synthetic) and inference (real-world) data distributions.