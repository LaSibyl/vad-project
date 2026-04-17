"""
CS 6140 — Voice Activity Detection
Deep Learning Model: Architecture + Training Demo
Author: Ziming
Role:   DL Model Design & Training

This script demonstrates:
  1. Synthetic data generation (stand-in for LibriSpeech + MUSAN)
  2. Log-Mel Spectrogram feature extraction
  3. Lightweight CNN VAD model (<500K parameters)
  4. Training loop with loss/accuracy output
  5. Augmentation strategy scaffold (no-aug / uniform / curriculum)

Run:  python vad_dl_demo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import time

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_MS      = 20          # 20 ms frames (matches proposal)
FRAME_LEN     = int(SAMPLE_RATE * FRAME_MS / 1000)   # 320 samples
N_MELS        = 40
N_FRAMES      = 5           # context window: 5 frames → 100 ms
BATCH_SIZE    = 64
EPOCHS        = 5           # short demo; real training would be ~20+
LR            = 1e-3

print("=" * 60)
print("CS 6140  |  VAD Deep Learning Demo")
print("=" * 60)
print(f"Config: {FRAME_MS}ms frames | {N_MELS} mel bins | seed={SEED}")
print()


# ── 1. Log-Mel Spectrogram Feature Extractor ─────────────────────────────────
def build_mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=512, n_mels=N_MELS):
    """
    Construct a mel filterbank matrix (numpy).
    Maps linear FFT bins → n_mels triangular mel filters.
    """
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
    return torch.tensor(fbank, dtype=torch.float32)   # (n_mels, n_fft//2+1)


class LogMelExtractor(nn.Module):
    """
    Converts a raw audio waveform into a Log-Mel Spectrogram.
    Pure PyTorch — no torchaudio dependency.

    Input : (batch, samples)  — single-channel audio frame
    Output: (batch, n_mels)   — one log-mel feature vector
    """
    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=512,
                 n_mels=N_MELS, frame_len=FRAME_LEN):
        super().__init__()
        self.n_fft     = n_fft
        self.frame_len = frame_len
        # Hann window
        self.register_buffer('window', torch.hann_window(frame_len))
        # Mel filterbank
        self.register_buffer('mel_fb', build_mel_filterbank(sample_rate, n_fft, n_mels))

    def forward(self, waveform):
        # waveform: (batch, samples)
        # Pad or trim to frame_len
        batch = waveform.shape[0]
        x = waveform[:, :self.frame_len]
        if x.shape[1] < self.frame_len:
            x = torch.nn.functional.pad(x, (0, self.frame_len - x.shape[1]))

        # STFT (manual via rfft)
        x_win  = x * self.window.unsqueeze(0)          # apply Hann window
        n_pad  = self.n_fft - self.frame_len
        x_pad  = torch.nn.functional.pad(x_win, (0, n_pad))
        spec   = torch.fft.rfft(x_pad, n=self.n_fft)   # (batch, n_fft//2+1)
        power  = spec.real ** 2 + spec.imag ** 2        # power spectrum

        # Apply mel filterbank
        mel    = torch.matmul(power, self.mel_fb.T)     # (batch, n_mels)

        # Log (clamp for numerical stability)
        log_mel = torch.log(mel.clamp(min=1e-9))
        return log_mel


print("── Feature Extractor ────────────────────────────────────")
extractor = LogMelExtractor()
dummy_wave = torch.randn(4, FRAME_LEN)
dummy_feat = extractor(dummy_wave)
print(f"Input waveform shape : {dummy_wave.shape}")
print(f"Log-Mel feature shape: {dummy_feat.shape}  (batch × {N_MELS} mel bins)")
print()


# ── 2. Lightweight CNN VAD Model ──────────────────────────────────────────────
class LightweightCNN_VAD(nn.Module):
    """
    Small 1-D CNN for frame-level VAD.
    Input : (batch, 1, n_mels × n_frames)  — flattened context window
    Output: (batch, 1)                     — speech probability

    Design constraints (from proposal):
      • <500K parameters
      • No look-ahead (causal)
      • CPU-only inference target
    """
    def __init__(self, n_mels=N_MELS, n_frames=N_FRAMES):
        super().__init__()
        in_len = n_mels * n_frames          # 40 × 5 = 200

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                # 200 → 100

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                # 100 → 50

            # Block 3
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),        # → 8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, n_mels × n_frames)
        x = x.unsqueeze(1)          # → (batch, 1, feature_len)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = LightweightCNN_VAD()
total_params = sum(p.numel() for p in model.parameters())
print("── Model Architecture ───────────────────────────────────")
print(model)
print()
print(f"Total parameters : {total_params:,}")
print(f"Constraint check : {'✓ PASS' if total_params < 500_000 else '✗ FAIL'} (<500K)")
print()


# ── 3. Synthetic Dataset (stand-in for LibriSpeech + MUSAN) ──────────────────
class SyntheticVADDataset(Dataset):
    """
    Generates synthetic speech/non-speech frames for demo purposes.
    In real training this is replaced by LibriSpeech + MUSAN mixing.

    Augmentation strategies (from proposal):
      strategy='none'       — no noise added
      strategy='uniform'    — random SNR from [0, 20] dB uniformly
      strategy='curriculum' — SNR starts easy (high) and decreases each epoch
    """
    def __init__(self, n_samples=2000, strategy='none', snr_db=None):
        self.n_samples = n_samples
        self.strategy  = strategy
        self.snr_db    = snr_db     # used by curriculum to pass current SNR
        self.feature_dim = N_MELS * N_FRAMES
        self._generate()

    def _mix_noise(self, signal, snr_db):
        """Add white noise at a given SNR level."""
        sig_power   = signal.pow(2).mean()
        noise       = torch.randn_like(signal)
        noise_power = noise.pow(2).mean()
        scale       = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
        return signal + scale * noise

    def _generate(self):
        self.X, self.y = [], []
        for _ in range(self.n_samples):
            label = random.randint(0, 1)

            if label == 1:  # speech: harmonic-like signal
                freq = random.uniform(80, 300)
                t    = torch.linspace(0, FRAME_MS / 1000 * N_FRAMES, N_MELS * N_FRAMES)
                feat = torch.sin(2 * np.pi * freq * t)
                feat = feat + 0.1 * torch.randn_like(feat)
            else:           # non-speech: noise only
                feat = 0.05 * torch.randn(N_MELS * N_FRAMES)

            # Apply augmentation strategy
            if self.strategy == 'uniform':
                snr = random.uniform(0, 20)
                feat = self._mix_noise(feat, snr)
            elif self.strategy == 'curriculum' and self.snr_db is not None:
                feat = self._mix_noise(feat, self.snr_db)

            self.X.append(feat)
            self.y.append(float(label))

        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):  return self.n_samples
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ── 4. Training Loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch).squeeze()
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        predicted   = (preds > 0.5).float()
        correct    += (predicted == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp = fp = fn = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).squeeze()
            loss  = criterion(preds, y_batch)

            total_loss += loss.item() * len(y_batch)
            predicted   = (preds > 0.5).float()
            correct    += (predicted == y_batch).sum().item()
            total      += len(y_batch)

            tp += ((predicted == 1) & (y_batch == 1)).sum().item()
            fp += ((predicted == 1) & (y_batch == 0)).sum().item()
            fn += ((predicted == 0) & (y_batch == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return total_loss / total, correct / total, f1


def run_training(strategy_name, snr_schedule=None):
    """
    Train model under a given augmentation strategy.
    snr_schedule: list of SNR values per epoch (for curriculum strategy).
    """
    print(f"\n{'═' * 60}")
    print(f"  Strategy: {strategy_name.upper()}")
    print(f"{'═' * 60}")

    device    = torch.device('cpu')   # CPU-only per proposal
    model_run = LightweightCNN_VAD().to(device)
    optimizer = optim.Adam(model_run.parameters(), lr=LR)
    criterion = nn.BCELoss()

    val_dataset = SyntheticVADDataset(500, strategy='none')
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    history = []
    for epoch in range(1, EPOCHS + 1):
        # Build training dataset for this epoch
        snr = snr_schedule[epoch - 1] if snr_schedule else None
        train_dataset = SyntheticVADDataset(
            2000, strategy=strategy_name, snr_db=snr
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model_run, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model_run, val_loader, criterion, device)
        elapsed = time.time() - t0

        snr_info = f"  SNR={snr:.0f}dB" if snr is not None else ""
        print(f"  Epoch {epoch}/{EPOCHS}{snr_info}  |  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"val_F1={val_f1:.3f}  [{elapsed:.1f}s]")
        history.append({'epoch': epoch, 'val_f1': val_f1, 'val_loss': val_loss})

    final_f1 = history[-1]['val_f1']
    print(f"\n  Final val F1 ({strategy_name}): {final_f1:.4f}")
    return history, model_run


# ── Run all three augmentation strategies (from proposal) ────────────────────
print("── Training: Strategy Comparison ───────────────────────")
print("  Strategies: ① No-Aug  ② Uniform SNR  ③ Curriculum SNR")
print()

# ① No augmentation (baseline DL)
hist_none, model_none = run_training('none')

# ② Uniform SNR augmentation (0–20 dB randomly per sample)
hist_uniform, model_uniform = run_training('uniform')

# ③ Curriculum augmentation: start easy (20 dB) → get harder (0 dB)
curriculum_snr = [20, 15, 10, 5, 0]   # one SNR per epoch
hist_curriculum, model_curriculum = run_training('curriculum', snr_schedule=curriculum_snr)


# ── 5. Summary Table ──────────────────────────────────────────────────────────
print()
print("── Results Summary ──────────────────────────────────────")
print(f"  {'Strategy':<18}  {'Final Val F1':>12}  {'Best Val F1':>11}")
print(f"  {'-'*18}  {'-'*12}  {'-'*11}")
for name, hist in [('No Augmentation', hist_none),
                   ('Uniform SNR',     hist_uniform),
                   ('Curriculum SNR',  hist_curriculum)]:
    final = hist[-1]['val_f1']
    best  = max(h['val_f1'] for h in hist)
    print(f"  {name:<18}  {final:>12.4f}  {best:>11.4f}")

print()
print("── Latency Benchmark (CPU) ──────────────────────────────")
model_curriculum.eval()
dummy_input = torch.randn(1, N_MELS * N_FRAMES)
N_RUNS = 1000
t0 = time.time()
with torch.no_grad():
    for _ in range(N_RUNS):
        _ = model_curriculum(dummy_input)
avg_ms = (time.time() - t0) / N_RUNS * 1000
print(f"  Avg inference time over {N_RUNS} frames: {avg_ms:.3f} ms/frame")
print(f"  Real-time constraint check : {'✓ PASS' if avg_ms < 20 else '✗ FAIL'} (must be <{FRAME_MS}ms)")

print()
print("── Next Steps ───────────────────────────────────────────")
print("  • Replace synthetic data with real LibriSpeech + MUSAN")
print("  • Finalize CNN vs. MLP architecture selection")
print("  • Extend training to 20+ epochs on full dataset")
print("  • Coordinate with Gengyuan on shared eval module format")
print()
print("Done.")