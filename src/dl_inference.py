import math
import numpy as np
import torch
import librosa
from pathlib import Path
from torch.utils.data import DataLoader

from vad_dl_demo import LightweightCNN_VAD, SyntheticVADDataset, LogMelExtractor

# ── Numpy-based feature extraction (matches build_real_dataset.py exactly) ───
# Duplicated here to keep dl_inference self-contained and avoid triggering
# build_real_dataset.py's module-level side effects on import.

SAMPLE_RATE = 16000
FRAME_LEN   = 320   # 20 ms @ 16 kHz
N_MELS      = 40
N_FRAMES    = 5
N_FFT       = 512


def _build_mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    def hz_to_mel(hz):  return 2595.0 * math.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel  = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = np.array([mel_to_hz(m) for m in mel_pts])
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbank


def _extract_log_mel(frame):
    """Extract log-mel features from a single 320-sample frame (numpy)."""
    window = np.hanning(FRAME_LEN)
    x = np.zeros(FRAME_LEN)
    x[:len(frame)] = frame[:FRAME_LEN]
    x = x * window

    x_padded = np.zeros(N_FFT)
    x_padded[:FRAME_LEN] = x
    spec  = np.fft.rfft(x_padded)
    power = spec.real ** 2 + spec.imag ** 2

    mel_fb  = _build_mel_filterbank()
    mel     = np.dot(power, mel_fb.T)
    log_mel = np.log(np.clip(mel, a_min=1e-9, a_max=None))
    return log_mel   # (N_MELS,)


def _extract_context_features(audio_segment):
    """
    Extract a (N_MELS * N_FRAMES,) feature vector from a 1600-sample segment.
    Identical to build_real_dataset.extract_context_features().
    """
    expected = FRAME_LEN * N_FRAMES   # 1600
    if len(audio_segment) < expected:
        audio_segment = np.pad(audio_segment, (0, expected - len(audio_segment)))
    else:
        audio_segment = audio_segment[:expected]

    features = []
    for i in range(N_FRAMES):
        frame = audio_segment[i * FRAME_LEN: (i + 1) * FRAME_LEN]
        features.append(_extract_log_mel(frame))

    return np.concatenate(features).astype(np.float32)  # (200,)


def run_dl_inference(
    strategy="none",
    n_samples=500,
    batch_size=64,
    snr_db=None,
    checkpoint_path=None,
):
    device = torch.device("cpu")

    dataset = SyntheticVADDataset(
        n_samples=n_samples,
        strategy=strategy,
        snr_db=snr_db
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LightweightCNN_VAD().to(device)

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Warning: no checkpoint loaded, using randomly initialized model.")

    model.eval()

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze()
            probs = torch.sigmoid(logits)   # model outputs logits; convert to probabilities
            preds = (probs > 0.5).float()

            y_true_all.extend(y_batch.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
            y_prob_all.extend(probs.cpu().numpy())

    return {
        "y_true": np.array(y_true_all).astype(int),
        "y_pred": np.array(y_pred_all).astype(int),
        "y_prob": np.array(y_prob_all),
        "strategy": strategy,
        "n_samples": n_samples,
    }


def load_audio_signal(audio_path):
    """Load audio as a raw float32 signal at 16 kHz."""
    signal, _ = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio: {len(signal)} samples ({len(signal)/16000:.2f}s)")
    return signal


def build_cnn_inputs(signal, hop=160, return_stats=False):
    """
    Convert raw audio signal → CNN feature matrix using the same pipeline
    as build_real_dataset.py (numpy FFT, non-overlapping frames per window).

    Slides a 1600-sample (100 ms) window over the signal with ``hop`` steps.
    Default hop=160 (10 ms) gives one prediction per energy-VAD frame.

    Returns:
        inputs: (T, 200) float32 feature matrix  [return_stats=False]
        (inputs, stats): tuple                   [return_stats=True]
    """
    window_len = FRAME_LEN * N_FRAMES   # 1600 samples
    positions  = range(0, len(signal) - window_len + 1, hop)

    inputs = np.array(
        [_extract_context_features(signal[s: s + window_len]) for s in positions],
        dtype=np.float32,
    )

    if return_stats:
        stats = {
            "min":    inputs.min(),
            "max":    inputs.max(),
            "mean":   inputs.mean(),
            "std":    inputs.std(),
            "median": np.median(inputs),
            "shape":  inputs.shape,
            "data":   inputs.flatten(),
        }
        return inputs, stats
    return inputs


def run_dl_on_audio(audio_path, checkpoint_path, return_feature_stats=False, normalize_inputs=False):
    """
    Run CNN inference on real audio using the same feature extraction
    pipeline as training (numpy FFT, non-overlapping frames per window).

    Args:
        audio_path:           path to audio file
        checkpoint_path:      path to saved model checkpoint
        return_feature_stats: if True, include feature statistics in result
        normalize_inputs:     ignored (kept for API compatibility; normalization
                              is no longer applied — use matched extractor instead)

    Returns:
        dict with y_prob, y_pred, and optionally feature_stats
    """
    device = torch.device("cpu")

    # 1. load raw signal
    signal = load_audio_signal(audio_path)

    # 2. extract features (same pipeline as training)
    X, feature_stats = build_cnn_inputs(signal, return_stats=True)
    print(f"Feature matrix: {X.shape}")

    # 3. apply the same z-score normalisation used at training time
    scaler_path = str(Path(checkpoint_path).parent / "feature_scaler.pt")
    if Path(scaler_path).exists():
        scaler = torch.load(scaler_path, map_location="cpu", weights_only=True)
        feat_mean = scaler["mean"].numpy()   # (200,)
        feat_std  = scaler["std"].numpy()    # (200,)
        X = (X - feat_mean) / (feat_std + 1e-6)
        print(f"Applied feature scaler (mean_avg={feat_mean.mean():.4f}, std_avg={feat_std.mean():.4f})")
    else:
        print(f"Warning: no feature_scaler.pt found at {scaler_path}; skipping normalisation")

    # 4. load model
    model = LightweightCNN_VAD().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. inference
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor).squeeze()
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs > 0.5).astype(int)

    result = {"y_prob": probs, "y_pred": preds}
    if return_feature_stats:
        result["feature_stats"] = feature_stats
    return result

