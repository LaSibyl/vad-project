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
    curriculum_ckpt = project_root / "checkpoints" / "cnn_curriculum.pt"

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

    print_metrics("CNN VAD - Synthetic Demo", metrics_dl)

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
    # Now with consistent framing (hop_len=160, matching Energy VAD)
    # Energy: 615 frames (hop=160, 50% overlap)
    # DL CNN: 615 frames → 611 after 5-frame sliding window
    # GT slice: [2:-2] of 615 → 611 
    dl_real = run_dl_on_audio(
        audio_path="data/sample.wav",
        checkpoint_path=str(curriculum_ckpt),
        return_feature_stats=True
    )
    
    real_stats = dl_real.get("feature_stats")
    if real_stats:
        print(f"\n📊 Real Audio Feature Stats:")
        print(f"  Min    : {real_stats['min']:>10.6f}")
        print(f"  Max    : {real_stats['max']:>10.6f}")
        print(f"  Mean   : {real_stats['mean']:>10.6f}")
        print(f"  Std    : {real_stats['std']:>10.6f}")
        print(f"  Median : {real_stats['median']:>10.6f}")
        
        # Interpret the difference
        print(f"\n🔍 Analysis:")
        range_syn = syn_stats['max'] - syn_stats['min']
        range_real = real_stats['max'] - real_stats['min']
        
        print(f"  Synthetic range  : [{syn_stats['min']:.2f}, {syn_stats['max']:.2f}] (span={range_syn:.2f})")
        print(f"  Real range       : [{real_stats['min']:.2f}, {real_stats['max']:.2f}] (span={range_real:.2f})")
        
        if range_real > range_syn * 2:
            print(f"  ⚠️  Real has {range_real/range_syn:.1f}x larger variation!")
        if abs(syn_stats['mean'] - real_stats['mean']) > syn_stats['std']:
            print(f"  ⚠️  Large mean shift: {real_stats['mean'] - syn_stats['mean']:.4f}")
        if real_stats['std'] < syn_stats['std'] / 2:
            print(f"  ⚠️  Real has much smaller variance (noise diversity problem)")

    gt_full = energy_result["y_true"]
    gt_aligned = gt_full[2:-2]
    
    print(f"\n── Length Check ──")
    print(f"  GT (full)    : {len(gt_full)}")
    print(f"  GT (aligned) : {len(gt_aligned)}")
    print(f"  DL preds     : {len(dl_real['y_pred'])}")
    
    if len(gt_aligned) != len(dl_real["y_pred"]):
        print(f"  ⚠️  WARNING: Length mismatch!")
    else:
        print(f"  ✓ Lengths match!")

    metrics_dl_real = compute_metrics(
        gt_aligned,
        dl_real["y_pred"]
    )
    
    print_metrics("CNN VAD - Real Audio (Unnormalized)", metrics_dl_real)
    
    # ===== P0 FIX TEST: Feature Normalization =====
    # 🧪 Quick hypothesis test: does per-window normalization fix the saturation?
    print("\n" + "=" * 68)
    print("P0 FIX ATTEMPT: Per-Window Feature Normalization")
    print("=" * 68)
    print("Testing: window = (window - mean) / (std + 1e-6)")
    print()
    
    dl_real_normalized = run_dl_on_audio(
        audio_path="data/sample.wav",
        checkpoint_path=str(curriculum_ckpt),
        return_feature_stats=False,
        normalize_inputs=True
    )
    
    metrics_dl_real_norm = compute_metrics(
        gt_aligned,
        dl_real_normalized["y_pred"]
    )
    
    print_metrics("CNN VAD - Real Audio (Normalized)", metrics_dl_real_norm)
    
    # Compare
    print(f"\n📊 Normalization Impact:")
    print(f"  Unnormalized: F1={metrics_dl_real['F1']:.4f}, FAR={metrics_dl_real['FAR']:.4f}, MISS={metrics_dl_real['MISS']:.4f}")
    print(f"  Normalized  : F1={metrics_dl_real_norm['F1']:.4f}, FAR={metrics_dl_real_norm['FAR']:.4f}, MISS={metrics_dl_real_norm['MISS']:.4f}")
    
    f1_improvement = ((metrics_dl_real_norm['F1'] - metrics_dl_real['F1']) / (metrics_dl_real['F1'] + 1e-8)) * 100
    print(f"  F1 change: {f1_improvement:+.1f}%")
    
    if metrics_dl_real_norm['FAR'] < metrics_dl_real['FAR']:
        print(f"\n✓ Normalization reduced FAR!")
    if metrics_dl_real_norm['MISS'] < 0.5:
        print(f"✓ MISS rate reasonable (not all missed)")
    
    # Use normalized results for rest of script
    print("\n👉 Using NORMALIZED results for further analysis...")

    # ===== DIAGNOSTIC: Threshold Sweep (on NORMALIZED inputs) =====
    print("\n" + "=" * 68)
    print("DIAGNOSTIC: Threshold Sweep on Real Audio (Normalized Inputs)")
    print("=" * 68)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    probs = dl_real_normalized["y_prob"]  # Use normalized version
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    print(f"{'Threshold':>10} {'F1':>10} {'FAR':>10} {'MISS':>10}")
    print("-" * 40)
    
    for t in thresholds:
        preds_t = (probs > t).astype(int)
        metrics_t = compute_metrics(gt_aligned, preds_t)
        f1 = metrics_t['F1']
        far = metrics_t['FAR']
        miss = metrics_t['MISS']
        
        marker = " ← BEST" if f1 > best_f1 else ""
        print(f"{t:>10.1f} {f1:>10.4f} {far:>10.4f} {miss:>10.4f}{marker}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"\nBest threshold: {best_threshold:.1f} (F1={best_f1:.4f})")
    
    # ===== DIAGNOSTIC: Probability Distribution =====
    print("\n" + "=" * 68)
    print("DIAGNOSTIC: Probability Distribution Statistics")
    print("=" * 68)
    
    prob_min = probs.min()
    prob_max = probs.max()
    prob_mean = probs.mean()
    prob_median = np.median(probs)
    
    print(f"Min probability    : {prob_min:.6f}")
    print(f"Max probability    : {prob_max:.6f}")
    print(f"Mean probability   : {prob_mean:.6f}")
    print(f"Median probability : {prob_median:.6f}")
    print(f"Std deviation      : {probs.std():.6f}")
    
    # Count how many probs are > 0.5
    above_threshold_count = (probs > 0.5).sum()
    print(f"\nFrames >  0.5: {above_threshold_count}/{len(probs)} ({100*above_threshold_count/len(probs):.1f}%)")
    print(f"Frames > {best_threshold:.1f}: {(probs > best_threshold).sum()}/{len(probs)}")
    
    # ===== DIAGNOSTIC: Plot probability distribution =====
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(probs, linewidth=0.8)
    plt.axhline(0.5, color='r', linestyle='--', label='Default threshold (0.5)', linewidth=1.5)
    plt.axhline(best_threshold, color='g', linestyle='--', label=f'Tuned threshold ({best_threshold:.1f})', linewidth=1.5)
    plt.axhline(prob_mean, color='b', linestyle=':', label=f'Mean ({prob_mean:.3f})', linewidth=1.5)
    plt.xlabel("Frame Index")
    plt.ylabel("Speech Probability")
    plt.title("CNN Speech Probabilities on Real Audio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(probs, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0.5, color='r', linestyle='--', label='Default (0.5)', linewidth=1.5)
    plt.axvline(best_threshold, color='g', linestyle='--', label=f'Tuned ({best_threshold:.1f})', linewidth=1.5)
    plt.xlabel("Speech Probability")
    plt.ylabel("Count")
    plt.title("Distribution of Probabilities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create outputs directory if it doesn't exist
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    plt.savefig(str(outputs_dir / "probability_distribution.png"), dpi=100, bbox_inches='tight')
    print(f"\n✓ Probability distribution plot saved to: outputs/probability_distribution.png")
    plt.show()
    
    # ===== FEATURE DISTRIBUTION HISTOGRAM COMPARISON =====
    if real_stats:
        print("\n" + "=" * 68)
        print("FEATURE DISTRIBUTION COMPARISON: Synthetic vs Real")
        print("=" * 68)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(syn_stats['data'], bins=50, alpha=0.6, label='Synthetic', edgecolor='black')
        plt.hist(real_stats['data'], bins=50, alpha=0.6, label='Real Audio', edgecolor='black')
        plt.xlabel("Feature Value")
        plt.ylabel("Count")
        plt.title("CNN Input Feature Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log scale version
        plt.subplot(1, 2, 2)
        plt.hist(syn_stats['data'], bins=50, alpha=0.6, label='Synthetic', edgecolor='black')
        plt.hist(real_stats['data'], bins=50, alpha=0.6, label='Real Audio', edgecolor='black')
        plt.xlabel("Feature Value")
        plt.ylabel("Count (log scale)")
        plt.title("CNN Input Feature Distribution (Log Scale)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(str(outputs_dir / "feature_distribution_comparison.png"), dpi=100, bbox_inches='tight')
        print(f"✓ Feature distribution comparison saved to: outputs/feature_distribution_comparison.png")
        plt.show()
    
    # ===== REAL RESULTS: Using tuned threshold =====
    print("\n" + "=" * 68)
    print(f"RESULTS: CNN VAD with Tuned Threshold ({best_threshold:.1f})")
    print("=" * 68)
    
    preds_tuned = (probs > best_threshold).astype(int)
    metrics_dl_real_tuned = compute_metrics(gt_aligned, preds_tuned)
    
    print_metrics(f"CNN VAD - Real Audio (Normalized + Threshold={best_threshold:.1f})", metrics_dl_real_tuned)
    
    print(f"\n📊 Summary of Real Audio Results:")
    print(f"  Unnormalized (t=0.5) : F1={metrics_dl_real['F1']:.4f}, FAR={metrics_dl_real['FAR']:.4f}, MISS={metrics_dl_real['MISS']:.4f}")
    print(f"  Normalized (t=0.5)   : F1={metrics_dl_real_norm['F1']:.4f}, FAR={metrics_dl_real_norm['FAR']:.4f}, MISS={metrics_dl_real_norm['MISS']:.4f}")
    print(f"  Normalized (t={best_threshold:.1f})   : F1={metrics_dl_real_tuned['F1']:.4f}, FAR={metrics_dl_real_tuned['FAR']:.4f}, MISS={metrics_dl_real_tuned['MISS']:.4f}")
    
    norm_improvement = ((metrics_dl_real_norm['F1'] - metrics_dl_real['F1']) / (metrics_dl_real['F1'] + 1e-8)) * 100
    tune_improvement = ((metrics_dl_real_tuned['F1'] - metrics_dl_real_norm['F1']) / (metrics_dl_real_norm['F1'] + 1e-8)) * 100
    
    print(f"\n  🔧 Normalization improvement (vs unnormalized): {norm_improvement:+.1f}%")
    print(f"  🎯 Tuning improvement (vs normalized):        {tune_improvement:+.1f}%")

    # ===== Summary table =====
    results = {
        "Energy Raw": metrics_energy_raw,
        "Energy Smoothed": metrics_energy_smooth,
        "CNN Synthetic": metrics_dl,
        "CNN Real Audio (Norm)": metrics_dl_real_norm,
    }
    print_comparison_table(results)

    print("\nNotes:")
    print("- Energy VAD: evaluated on real sample.wav with manual ground truth")
    print("- CNN Synthetic: evaluated on synthetic data generated by DL pipeline")
    print("- CNN Real Audio: evaluated on real audio with feature normalization applied")
    print("- Normalization is a P0 fix to address feature distribution mismatch")
    print()
    print("CNN Real Audio Probabilities (first 10):", dl_real_normalized["y_prob"][:10])
    print("CNN Real Audio Predictions (first 10) :", dl_real_normalized["y_pred"][:10])
    print("Positive predictions:", dl_real_normalized["y_pred"].sum(), f"/{len(dl_real_normalized['y_pred'])}")


if __name__ == "__main__":
    main()