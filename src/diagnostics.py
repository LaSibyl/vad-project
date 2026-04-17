import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from vad_dl_demo import SyntheticVADDataset
from dl_inference import build_cnn_inputs, load_audio_frames

# Ensure outputs directory exists
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def visualize_synthetic_feature_space(n_samples=2000, strategy="none", snr_db=None, save=True):
    dataset = SyntheticVADDataset(
        n_samples=n_samples,
        strategy=strategy,
        snr_db=snr_db
    )

    X_syn = dataset.X.numpy()
    y_syn = dataset.y.numpy().astype(int)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_syn)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_pca[y_syn == 0, 0],
        X_pca[y_syn == 0, 1],
        label="non-speech",
        alpha=0.5,
        s=12
    )
    plt.scatter(
        X_pca[y_syn == 1, 0],
        X_pca[y_syn == 1, 1],
        label="speech",
        alpha=0.5,
        s=12
    )
    plt.legend()
    plt.title(f"Synthetic Feature Space ({strategy})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        filename = OUTPUT_DIR / f"01_synthetic_feature_space_{strategy}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
    
    plt.show()


def compute_cluster_separability(n_samples=2000, strategy="none", snr_db=None):
    """
    Compute metrics to quantify speech vs non-speech separability in synthetic data.
    
    Higher silhouette_score (closer to 1) = better separated
    Lower davies_bouldin_score = better separated
    """
    dataset = SyntheticVADDataset(
        n_samples=n_samples,
        strategy=strategy,
        snr_db=snr_db
    )

    X_syn = dataset.X.numpy()
    y_syn = dataset.y.numpy().astype(int)

    # Silhouette Score (range [-1, 1], higher is better)
    sil_score = silhouette_score(X_syn, y_syn)
    
    # Davies-Bouldin Index (lower is better)
    db_score = davies_bouldin_score(X_syn, y_syn)

    print(f"\n=== Cluster Separability Analysis ({strategy}) ===")
    print(f"Silhouette Score       : {sil_score:.4f}  (range [-1,1], higher better)")
    print(f"Davies-Bouldin Index   : {db_score:.4f}  (lower better)")
    
    if sil_score > 0.5:
        print("✓ GOOD: Clusters are well separated")
    elif sil_score > 0.3:
        print("⚠️  OK: Clusters are moderately separated")
    else:
        print("❌ POOR: Clusters are overlapping (may cause shortcut learning!)")
    
    return {"silhouette": sil_score, "davies_bouldin": db_score}


def visualize_real_audio_feature_space(audio_path="data/sample.wav", save=True):
    """
    Visualize real audio feature distribution (for comparison with synthetic).
    """
    try:
        frames = load_audio_frames(audio_path)
        X_real, stats = build_cnn_inputs(frames, return_stats=True)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_real)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=12, color='orange')
        plt.title(f"Real Audio Feature Space")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = OUTPUT_DIR / "02_real_audio_feature_space.png"
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        plt.show()
        
        print(f"\n=== Real Audio Feature Stats ===")
        print(f"Samples       : {X_real.shape[0]}")
        print(f"Feature dim   : {X_real.shape[1]}")
        print(f"Min           : {stats['min']:.4f}")
        print(f"Max           : {stats['max']:.4f}")
        print(f"Mean          : {stats['mean']:.4f}")
        print(f"Std           : {stats['std']:.4f}")
        
    except Exception as e:
        print(f"Error loading real audio: {e}")


def compare_synthetic_vs_real(audio_path="data/sample.wav", n_samples_syn=2000, save=True):
    """
    Side-by-side comparison of synthetic vs real feature distributions.
    """
    # Synthetic
    dataset = SyntheticVADDataset(n_samples=n_samples_syn, strategy="none")
    X_syn = dataset.X.numpy()
    y_syn = dataset.y.numpy().astype(int)
    
    # Real
    try:
        frames = load_audio_frames(audio_path)
        X_real, _ = build_cnn_inputs(frames, return_stats=True)
    except Exception as e:
        print(f"Error loading real audio: {e}")
        return
    
    # PCA on synthetic
    pca_syn = PCA(n_components=2)
    X_syn_pca = pca_syn.fit_transform(X_syn)
    
    # PCA on real (fit separately)
    pca_real = PCA(n_components=2)
    X_real_pca = pca_real.fit_transform(X_real)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Synthetic
    ax1.scatter(
        X_syn_pca[y_syn==0, 0], X_syn_pca[y_syn==0, 1],
        label="non-speech", alpha=0.5, s=12
    )
    ax1.scatter(
        X_syn_pca[y_syn==1, 0], X_syn_pca[y_syn==1, 1],
        label="speech", alpha=0.5, s=12
    )
    ax1.set_title("Synthetic Data (labeled)")
    ax1.set_xlabel(f"PC1 ({pca_syn.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca_syn.explained_variance_ratio_[1]:.1%})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Real
    ax2.scatter(X_real_pca[:, 0], X_real_pca[:, 1], alpha=0.6, s=12, color='orange')
    ax2.set_title(f"Real Audio (unlabeled)")
    ax2.set_xlabel(f"PC1 ({pca_real.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca_real.explained_variance_ratio_[1]:.1%})")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = OUTPUT_DIR / "03_synthetic_vs_real_comparison.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
    
    plt.show()
    
    print(f"\n=== Synthetic vs Real Comparison ===")
    print(f"Synthetic samples  : {X_syn.shape[0]}")
    print(f"Real samples       : {X_real.shape[0]}")
    print(f"PCA Var explained (Synthetic) : PC1={pca_syn.explained_variance_ratio_[0]:.2%}, PC2={pca_syn.explained_variance_ratio_[1]:.2%}")
    print(f"PCA Var explained (Real)      : PC1={pca_real.explained_variance_ratio_[0]:.2%}, PC2={pca_real.explained_variance_ratio_[1]:.2%}")


if __name__ == "__main__":
    print("=" * 60)
    print("VAD Model Diagnostics - Shortcut Learning Analysis")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Step 1: Visualize synthetic data
    visualize_synthetic_feature_space()
    
    # Step 2: Compute separability
    compute_cluster_separability()
    
    # Step 3: Visualize real audio
    visualize_real_audio_feature_space()
    
    # Step 4: Compare
    compare_synthetic_vs_real()
    
    print("\n" + "=" * 60)
    print("✓ All diagnostics complete!")
    print(f"📊 Plots saved to: {OUTPUT_DIR}")
    print("=" * 60)