from pathlib import Path
import numpy as np
import torch
from evaluation import compute_metrics, print_metrics
from energy_vad_demo import run_energy_vad
from dl_inference import run_dl_inference, run_dl_on_audio
from vad_dl_demo import SyntheticVADDataset


def print_comparison_table(results):
    print("\n" + "=" * 68)
    print("VAD Method Comparison")
    print("=" * 68)
    print(f"{'Method':<20} {'F1':>10} {'FAR':>10} {'MISS':>10} {'FP':>8} {'FN':>8}")
    print("-" * 68)

    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['F1']:>10.4f} "
            f"{metrics['FAR']:>10.4f} "
            f"{metrics['MISS']:>10.4f} "
            f"{metrics['FP']:>8d} "
            f"{metrics['FN']:>8d}"
        )


def main():
    print("=" * 68)
    print("CS6140 Voice Activity Detection - Unified Comparison")
    print("=" * 68)

    project_root = Path(__file__).resolve().parent.parent
    curriculum_ckpt = project_root / "checkpoints" / "cnn_real_data.pt"

    # ===== Energy baseline =====
    energy_result = run_energy_vad(return_details=True, plot=False)

    metrics_energy_raw = compute_metrics(
        energy_result["y_true"],
        energy_result["y_pred_raw"]
    )
    metrics_energy_smooth = compute_metrics(
        energy_result["y_true"],
        energy_result["y_pred_smooth"]
    )

    print_metrics("Energy VAD - Raw", metrics_energy_raw)
    print_metrics("Energy VAD - Smoothed", metrics_energy_smooth)

    # ===== DL model =====
    # Note:
    # current DL side uses synthetic dataset and its own labels.
    # so this is a pipeline integration demo, not yet a fully matched
    # apples-to-apples comparison on the same real sample.
    dl_result = run_dl_inference(
        strategy="none",
        n_samples=500,
        batch_size=64,
        checkpoint_path=str(curriculum_ckpt),
    )

    metrics_dl = compute_metrics(
        dl_result["y_true"],
        dl_result["y_pred"]
    )

    print_metrics("CNN VAD - Real Audio Trained", metrics_dl)

    # ===== FEATURE DISTRIBUTION ANALYSIS (Step 1) =====
    print("\n" + "=" * 68)
    print("STEP 1: Feature Distribution Comparison")
    print("=" * 68)
    
    # Get synthetic feature stats
    syn_dataset = SyntheticVADDataset(n_samples=2000, strategy='none')
    X_syn = syn_dataset.X[:1000]
    syn_stats = {
        "min": X_syn.min().item(),
        "max": X_syn.max().item(), 
        "mean": X_syn.mean().item(),
        "std": X_syn.std().item(),
        "median": torch.median(X_syn).item(),
        "data": X_syn.cpu().numpy().flatten()
    }
    
    print(f"\n📊 Synthetic Data Feature Stats:")
    print(f"  Min    : {syn_stats['min']:>10.6f}")
    print(f"  Max    : {syn_stats['max']:>10.6f}")
    print(f"  Mean   : {syn_stats['mean']:>10.6f}")
    print(f"  Std    : {syn_stats['std']:>10.6f}")
    print(f"  Median : {syn_stats['median']:>10.6f}")

    # ===== DL model on real audio =====
    # Uses the unified numpy-based feature extractor (matching training pipeline).
    # hop=160 → one prediction per 10 ms, same resolution as Energy VAD.
    dl_real = run_dl_on_audio(
        audio_path="data/sample.wav",
        checkpoint_path=str(curriculum_ckpt),
        return_feature_stats=True
    )

    real_stats = dl_real.get("feature_stats")
    if real_stats:
        print(f"\n📊 Real Audio Feature Stats (unified extractor):")
        print(f"  Min    : {real_stats['min']:>10.6f}")
        print(f"  Max    : {real_stats['max']:>10.6f}")
        print(f"  Mean   : {real_stats['mean']:>10.6f}")
        print(f"  Std    : {real_stats['std']:>10.6f}")
        print(f"  Median : {real_stats['median']:>10.6f}")

        print(f"\n🔍 Distribution comparison (should now be aligned):")
        print(f"  Train data mean : {-3.66:.2f}  std : {4.18:.2f}")
        print(f"  Inference mean  : {real_stats['mean']:.2f}  std : {real_stats['std']:.2f}")

    # GT alignment: DL window i (hop=160, window=1600) is centred at frame i+4
    # so we shift GT by 4 at the start and trim to match prediction count.
    gt_full    = energy_result["y_true"]
    n_preds    = len(dl_real["y_pred"])
    gt_aligned = gt_full[4: 4 + n_preds]

    print(f"\n── Length Check ──")
    print(f"  GT (full)    : {len(gt_full)}")
    print(f"  GT (aligned) : {len(gt_aligned)}")
    print(f"  DL preds     : {n_preds}")

    if len(gt_aligned) != n_preds:
        print(f"  ⚠️  WARNING: Length mismatch! GT slice may be shorter than preds.")
        n_preds    = min(n_preds, len(gt_aligned))
        gt_aligned = gt_aligned[:n_preds]
        dl_real["y_pred"] = dl_real["y_pred"][:n_preds]
        dl_real["y_prob"] = dl_real["y_prob"][:n_preds]
    else:
        print(f"  ✓ Lengths match!")

    metrics_dl_real = compute_metrics(gt_aligned, dl_real["y_pred"])
    print_metrics("CNN VAD - Real Audio (unified extractor)", metrics_dl_real)

    # ===== Threshold sweep =====
    print("\n" + "=" * 68)
    print("DIAGNOSTIC: Threshold Sweep on Real Audio")
    print("=" * 68)

    probs = dl_real["y_prob"]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_threshold, best_f1 = 0.5, 0.0

    print(f"{'Threshold':>10} {'F1':>10} {'FAR':>10} {'MISS':>10}")
    print("-" * 40)
    for t in thresholds:
        preds_t   = (probs > t).astype(int)
        metrics_t = compute_metrics(gt_aligned, preds_t)
        marker    = " ← BEST" if metrics_t["F1"] > best_f1 else ""
        print(f"{t:>10.1f} {metrics_t['F1']:>10.4f} {metrics_t['FAR']:>10.4f} {metrics_t['MISS']:>10.4f}{marker}")
        if metrics_t["F1"] > best_f1:
            best_f1, best_threshold = metrics_t["F1"], t

    print(f"\nBest threshold: {best_threshold:.1f} (F1={best_f1:.4f})")

    # ===== Probability distribution plot =====
    print("\n" + "=" * 68)
    print("DIAGNOSTIC: Probability Distribution")
    print("=" * 68)
    print(f"Min  : {probs.min():.6f}")
    print(f"Max  : {probs.max():.6f}")
    print(f"Mean : {probs.mean():.6f}")
    print(f"Std  : {probs.std():.6f}")
    print(f"Frames > 0.5 : {(probs > 0.5).sum()}/{len(probs)} ({100*(probs>0.5).mean():.1f}%)")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(probs, linewidth=0.8)
    plt.axhline(0.5, color="r", linestyle="--", label="Default (0.5)", linewidth=1.5)
    if best_threshold != 0.5:
        plt.axhline(best_threshold, color="g", linestyle="--",
                    label=f"Best ({best_threshold:.1f})", linewidth=1.5)
    plt.xlabel("Frame Index"); plt.ylabel("Speech Probability")
    plt.title("CNN Speech Probabilities on Real Audio"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(probs, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(0.5, color="r", linestyle="--", label="Default (0.5)", linewidth=1.5)
    plt.xlabel("Speech Probability"); plt.ylabel("Count")
    plt.title("Distribution of Probabilities"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    plt.savefig(str(outputs_dir / "probability_distribution.png"), dpi=100, bbox_inches="tight")
    print(f"\n✓ Plot saved to: outputs/probability_distribution.png")
    plt.show()

    # ===== Summary table =====
    results = {
        "Energy Raw":          metrics_energy_raw,
        "Energy Smoothed":     metrics_energy_smooth,
        "CNN Synthetic":       metrics_dl,
        "CNN Real Audio":      metrics_dl_real,
    }
    print_comparison_table(results)

    print("\nNotes:")
    print("- Energy VAD: evaluated on real sample.wav with manual ground truth")
    print("- CNN Synthetic: evaluated on synthetic data (DL pipeline, separate labels)")
    print("- CNN Real Audio: unified numpy extractor matching training pipeline")
    print()
    print("CNN Real Audio Probabilities (first 10):", dl_real["y_prob"][:10])
    print("CNN Real Audio Predictions (first 10) :", dl_real["y_pred"][:10])
    print("Positive predictions:", dl_real["y_pred"].sum(), f"/{len(dl_real['y_pred'])}")


if __name__ == "__main__":
    main()