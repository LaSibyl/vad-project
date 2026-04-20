#!/usr/bin/env python
"""
Real Audio Training Script
Trains CNN VAD model on LibriSpeech + MUSAN real audio data instead of synthetic.

Key differences from vad_dl_demo.py:
- Uses RealVADDataset instead of SyntheticVADDataset
- Simpler training (no augmentation strategies comparison, just one model)
- Compares real-audio trained model against Energy VAD

Run: python train_on_real_data.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if __file__.endswith('.py') else Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import from vad_dl_demo and other modules
from vad_dl_demo import (
    SAMPLE_RATE, FRAME_MS, FRAME_LEN, N_MELS, N_FRAMES,
    LR, BATCH_SIZE, EPOCHS, PROJECT_ROOT, CHECKPOINT_DIR,
    LogMelExtractor, LightweightCNN_VAD, evaluate
)
from build_real_dataset import RealVADDataset

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

EPOCHS_REAL = 20  # More epochs for real data
LR_REAL = 1e-4    # Slightly lower LR for more stable training
MODEL_NAME = "cnn_real_data.pt"

# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. criterion should be BCEWithLogitsLoss."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch).squeeze()

        # Handle batch size = 1
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)

        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        predicted = (logits > 0.0).float()   # logit > 0 ↔ sigmoid > 0.5
        correct += (predicted == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def main():
    print("=" * 70)
    print("CNN VAD Training on Real Audio (LibriSpeech + MUSAN)")
    print("=" * 70)
    print(f"Config: LR={LR_REAL}, Epochs={EPOCHS_REAL}, Batch={BATCH_SIZE}")
    print()

    # Ensure checkpoint dir exists
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cpu')  # CPU-only per proposal
    print(f"Device: {device}")
    print()

    # Load real datasets
    print("── Loading Real Audio Datasets ──────────────────────────")
    try:
        train_dataset = RealVADDataset(split='train')
        val_dataset = RealVADDataset(split='val')
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Run build_real_dataset.py first!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Valid: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print()

    # Initialize model
    print("── Building Model ───────────────────────────────────────")
    model = LightweightCNN_VAD(n_mels=N_MELS, n_frames=N_FRAMES).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: LightweightCNN_VAD")
    print(f"  Parameters: {n_params:,}")
    if n_params > 500000:
        print(f"  ⚠ Warning: exceeds 500K parameter constraint!")
    else:
        print(f"  ✓ Within 500K parameter constraint")
    print()

    # Class balance analysis
    n_pos = train_dataset.y.sum().item()
    n_neg = len(train_dataset.y) - n_pos
    ratio = n_pos / max(n_neg, 1)
    print(f"  Speech (pos): {int(n_pos)}, Non-speech (neg): {int(n_neg)}")
    print(f"  Speech ratio: {n_pos / (n_pos + n_neg):.1%} — using no class weighting (mild imbalance)")
    print()

    # Optimizer & criterion
    # BCEWithLogitsLoss: numerically stable, no sigmoid needed in model.
    # No pos_weight: 2:1 imbalance is mild; letting the model learn the natural prior
    # yields better-calibrated probabilities and a 0.5 threshold that actually works.
    optimizer = optim.Adam(model.parameters(), lr=LR_REAL)
    criterion = nn.BCEWithLogitsLoss()
    eval_criterion = nn.BCEWithLogitsLoss()

    # Training loop
    print("── Training ─────────────────────────────────────────────")
    history = []
    best_f1 = 0

    for epoch in range(1, EPOCHS_REAL + 1):
        t0 = time.time()

        # Train
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, eval_criterion, device)

        elapsed = time.time() - t0
        history.append({'epoch': epoch, 'tr_loss': tr_loss, 'tr_acc': tr_acc,
                       'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})

        # Track best F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch

        print(f"  Epoch {epoch:2d}/{EPOCHS_REAL}  |  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  |  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_F1={val_f1:.3f}  "
              f"[{elapsed:.1f}s]")

    print()
    print(f"Best validation F1: {best_f1:.4f} (epoch {best_epoch})")
    print()

    # Save model
    print("── Saving Model ─────────────────────────────────────────")
    checkpoint_path = CHECKPOINT_DIR / MODEL_NAME
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  ✓ Saved: {checkpoint_path}")
    print()

    # Print final summary
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Evaluate this model on sample.wav with compare_models.py")
    print("  2. Compare F1 score against Energy VAD baseline")
    print("  3. Adjust hyperparameters if needed and retrain")
    print("  4. If performance is good, commit as new milestone")
    print()


if __name__ == "__main__":
    main()