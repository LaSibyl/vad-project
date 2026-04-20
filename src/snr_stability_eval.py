#!/usr/bin/env python
"""
SNR Stability Evaluation
========================
Evaluates Energy VAD and CNN VAD across a range of SNR levels (0–20 dB)
to directly address the research question:

  "Under real-time streaming constraints, which VAD approach is more
   stable across noise conditions?"

Outputs
-------
- outputs/snr_stability.png  — F1 vs SNR curve for both methods
- Console table: F1, FAR, MISS at each SNR + frame-level latency

Usage
-----
    python src/snr_stability_eval.py
"""

import sys
import time
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.signal import medfilt

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation import compute_metrics
from dl_inference import load_audio_signal, build_cnn_inputs
from vad_dl_demo import LightweightCNN_VAD

# ── config ────────────────────────────────────────────────────────────────────
SAMPLE_WAV   = PROJECT_ROOT / "data" / "sample.wav"
MUSAN_NOISE  = PROJECT_ROOT / "data" / "raw" / "musan" / "noise"
CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "cnn_real_data.pt"
SCALER       = PROJECT_ROOT / "checkpoints" / "feature_scaler.pt"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
SAMPLE_RATE  = 16000
FRAME_LEN    = int(0.025 * SAMPLE_RATE)   # 400 samples — energy VAD frame
HOP_LEN      = int(0.010 * SAMPLE_RATE)   # 160 samples — 10 ms hop

SNR_LEVELS   = [0, 5, 10, 15, 20]         # dB

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix speech and noise to achieve a target SNR (dB)."""
    # Tile noise to match speech length
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(speech)]

    speech_power = np.mean(speech ** 2) + 1e-10
    noise_power  = np.mean(noise  ** 2) + 1e-10
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    return speech + scale * noise


def load_noise_file(musan_noise_dir: Path) -> np.ndarray:
    """Load a single MUSAN noise file (first .wav found)."""
    for root, _, files in os.walk(musan_noise_dir):
        for f in files:
            if f.endswith(".wav"):
                audio, _ = librosa.load(os.path.join(root, f), sr=SAMPLE_RATE, mono=True)
                if len(audio) > SAMPLE_RATE * 2:   # at least 2 s
                    return audio
    raise RuntimeError(f"No suitable noise file found in {musan_noise_dir}")


def build_gt(signal: np.ndarray) -> np.ndarray:
    """
    Reconstruct ground-truth labels for the mixed signal using the same
    time-based rule as energy_vad_demo.py (speech 2.6 s – 4.15 s).
    """
    n_frames = (len(signal) - FRAME_LEN) // HOP_LEN + 1
    frame_times = np.arange(n_frames) * HOP_LEN / SAMPLE_RATE
    gt = np.zeros(n_frames, dtype=int)
    gt[(frame_times >= 2.6) & (frame_times <= 4.15)] = 1
    return gt


# ── Energy VAD (signal-level, no file I/O) ───────────────────────────────────

def run_energy_vad_on_signal(signal: np.ndarray):
    """Run energy VAD on a raw signal array. Returns (gt, pred_smooth)."""
    n_frames = (len(signal) - FRAME_LEN) // HOP_LEN + 1
    frames = np.array([
        signal[i * HOP_LEN: i * HOP_LEN + FRAME_LEN]
        for i in range(n_frames)
    ])
    log_energy = np.log(np.sum(frames ** 2, axis=1) + 1e-10)
    threshold  = np.mean(log_energy)
    vad_raw    = (log_energy > threshold).astype(int)
    vad_smooth = medfilt(vad_raw, kernel_size=5).astype(int)
    gt         = build_gt(signal)
    # align lengths
    n = min(len(gt), len(vad_smooth))
    return gt[:n], vad_smooth[:n]


# ── CNN VAD ───────────────────────────────────────────────────────────────────

def load_cnn_model():
    model = LightweightCNN_VAD()
    model.load_state_dict(
        torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    )
    model.eval()

    scaler     = torch.load(str(SCALER), map_location="cpu", weights_only=True)
    feat_mean  = scaler["mean"].numpy()
    feat_std   = scaler["std"].numpy() + 1e-6
    return model, feat_mean, feat_std


def run_cnn_vad_on_signal(signal: np.ndarray, model, feat_mean, feat_std):
    """Run CNN VAD on a raw signal array. Returns (gt_aligned, preds, probs)."""
    X, _ = build_cnn_inputs(signal, return_stats=True)
    X    = (X - feat_mean) / feat_std

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs > 0.5).astype(int)

    gt_full    = build_gt(signal)
    n_preds    = len(preds)
    gt_aligned = gt_full[4: 4 + n_preds]

    n = min(len(gt_aligned), n_preds)
    return gt_aligned[:n], preds[:n], probs[:n]


# ── Latency measurement ───────────────────────────────────────────────────────

def measure_latency(signal: np.ndarray, model, feat_mean, feat_std, n_runs: int = 10):
    """
    Measure per-frame inference latency for one 100-ms CNN window (1 prediction).
    Returns mean ms/frame over n_runs.
    """
    seg = signal[:1600]                            # one context window
    X   = build_cnn_inputs(seg, hop=1600)          # (1, 200)
    X   = (X - feat_mean) / feat_std
    t   = torch.tensor(X, dtype=torch.float32)

    # warm-up
    with torch.no_grad():
        model(t)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(t)
        times.append((time.perf_counter() - start) * 1000)   # ms

    return float(np.mean(times)), float(np.std(times))


def measure_energy_latency(signal: np.ndarray, n_runs: int = 10):
    """Measure per-frame Energy VAD latency (one frame = 25 ms window)."""
    frame = signal[:FRAME_LEN]

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        energy = np.sum(frame ** 2)
        _ = np.log(energy + 1e-10) > -5.0   # threshold check
        times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times)), float(np.std(times))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("SNR Stability Evaluation")
    print("=" * 68)

    # Load clean speech
    print(f"\nLoading speech: {SAMPLE_WAV.name}")
    speech, _ = librosa.load(str(SAMPLE_WAV), sr=SAMPLE_RATE, mono=True)

    # Load noise
    print(f"Loading noise from MUSAN...")
    noise = load_noise_file(MUSAN_NOISE)
    print(f"  Speech: {len(speech)/SAMPLE_RATE:.2f}s  |  Noise: {len(noise)/SAMPLE_RATE:.2f}s")

    # Load CNN model
    print(f"\nLoading CNN checkpoint: {CHECKPOINT.name}")
    model, feat_mean, feat_std = load_cnn_model()
    print("  ✓ Model loaded")

    # ── SNR sweep ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"{'SNR (dB)':>10} | {'Energy F1':>10} {'Energy FAR':>11} | "
          f"{'CNN F1':>8} {'CNN FAR':>9}")
    print(f"{'─'*68}")

    energy_f1s, cnn_f1s   = [], []
    energy_fars, cnn_fars = [], []
    energy_misses, cnn_misses = [], []

    for snr in SNR_LEVELS:
        mixed = mix_at_snr(speech, noise, snr)

        # Energy VAD
        gt_e, pred_e = run_energy_vad_on_signal(mixed)
        m_e = compute_metrics(gt_e, pred_e)

        # CNN VAD
        gt_c, pred_c, _ = run_cnn_vad_on_signal(mixed, model, feat_mean, feat_std)
        m_c = compute_metrics(gt_c, pred_c)

        energy_f1s.append(m_e["F1"]);   energy_fars.append(m_e["FAR"])
        energy_misses.append(m_e["MISS"])
        cnn_f1s.append(m_c["F1"]);      cnn_fars.append(m_c["FAR"])
        cnn_misses.append(m_c["MISS"])

        print(f"{snr:>10} | {m_e['F1']:>10.4f} {m_e['FAR']:>11.4f} | "
              f"{m_c['F1']:>8.4f} {m_c['FAR']:>9.4f}")

    # Summary statistics
    print(f"\n{'─'*68}")
    print(f"{'':>10}   {'Energy VAD':^24} | {'CNN VAD':^20}")
    print(f"{'Metric':>10}   {'Mean':>8} {'Std':>8} {'Min':>6} | "
          f"{'Mean':>8} {'Std':>8} {'Min':>6}")
    print(f"{'─'*68}")

    for label, e_vals, c_vals in [
        ("F1",   energy_f1s,    cnn_f1s),
        ("FAR",  energy_fars,   cnn_fars),
        ("MISS", energy_misses, cnn_misses),
    ]:
        e_arr = np.array(e_vals)
        c_arr = np.array(c_vals)
        print(f"{label:>10}   "
              f"{e_arr.mean():>8.4f} {e_arr.std():>8.4f} {e_arr.min():>6.4f} | "
              f"{c_arr.mean():>8.4f} {c_arr.std():>8.4f} {c_arr.min():>6.4f}")

    # ── Latency ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("Frame-level Inference Latency (CPU)")
    print(f"{'─'*68}")

    e_lat, e_std = measure_energy_latency(speech)
    c_lat, c_std = measure_cnn_latency(speech, model, feat_mean, feat_std)

    print(f"  Energy VAD : {e_lat:.4f} ± {e_std:.4f} ms/frame")
    print(f"  CNN VAD    : {c_lat:.4f} ± {c_std:.4f} ms/frame")
    print(f"  CNN/Energy speed ratio: {c_lat/e_lat:.1f}×")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("VAD Stability Across SNR Levels", fontsize=13, fontweight="bold")

    # F1 vs SNR
    ax = axes[0]
    ax.plot(SNR_LEVELS, energy_f1s, "o-", color="#2196F3", label="Energy VAD (smoothed)", linewidth=2)
    ax.plot(SNR_LEVELS, cnn_f1s,    "s-", color="#F44336", label="CNN VAD",               linewidth=2)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("F1 Score",  fontsize=11)
    ax.set_title("F1 Score vs SNR", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(SNR_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FAR vs SNR
    ax = axes[1]
    ax.plot(SNR_LEVELS, energy_fars, "o-", color="#2196F3", label="Energy VAD (smoothed)", linewidth=2)
    ax.plot(SNR_LEVELS, cnn_fars,    "s-", color="#F44336", label="CNN VAD",               linewidth=2)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("False Alarm Rate", fontsize=11)
    ax.set_title("False Alarm Rate vs SNR", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(SNR_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "snr_stability.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved: {out_path}")
    plt.close()


def measure_cnn_latency(signal, model, feat_mean, feat_std, n_runs=50):
    return measure_latency(signal, model, feat_mean, feat_std, n_runs)


if __name__ == "__main__":
    main()
