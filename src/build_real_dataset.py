#!/usr/bin/env python
"""
Real Audio Dataset Builder for VAD
Loads LibriSpeech + MUSAN and creates training/validation datasets.

Features:
- Loads .flac files from LibriSpeech train-clean-5 and dev-clean-2
- Loads noise from MUSAN
- Generates mixed audio samples with variable SNR
- Caches extracted features to disk for faster loading
- Returns same format as SyntheticVADDataset for drop-in replacement
"""

import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import pickle
import random
from tqdm import tqdm
import soundfile as sf

# Config (must match vad_dl_demo.py)
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_LEN = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320 samples
N_MELS = 40
N_FRAMES = 5
FEATURE_DIM = N_MELS * N_FRAMES  # 200

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Library paths
LIBRISPEECH_TRAIN = DATA_RAW_DIR / "LibriSpeech" / "train-clean-5"
LIBRISPEECH_DEV = DATA_RAW_DIR / "LibriSpeech" / "dev-clean-2"
MUSAN_NOISE_DIR = DATA_RAW_DIR / "musan" / "noise"

print("=" * 70)
print("REAL AUDIO DATASET BUILDER")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")
print(f"Data raw dir: {DATA_RAW_DIR}")
print(f"Data processed dir: {DATA_PROCESSED_DIR}")
print()

# Ensure processed dir exists
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Audio Loading & Feature Extraction (matching vad_dl_demo.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=512, n_mels=N_MELS):
    """Build mel filterbank (same as vad_dl_demo.py)"""
    import math
    
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


def extract_log_mel_features(audio_frame, sample_rate=SAMPLE_RATE, n_fft=512, n_mels=N_MELS):
    """
    Extract log-mel spectrogram from audio frame.
    Mirrors vad_dl_demo.py LogMelExtractor.
    
    Args:
        audio_frame: (samples,) or (batch, samples)
        
    Returns:
        (n_mels,) or (batch, n_mels) log-mel features
    """
    # Handle single frame
    single = False
    if audio_frame.ndim == 1:
        audio_frame = audio_frame[np.newaxis, :]
        single = True
    
    batch = audio_frame.shape[0]
    window = np.hanning(min(FRAME_LEN, audio_frame.shape[1]))
    
    # Pad/trim to frame_len
    x = np.zeros((batch, FRAME_LEN))
    for i in range(batch):
        frame = audio_frame[i]
        if len(frame) <= FRAME_LEN:
            x[i, :len(frame)] = frame
        else:
            x[i] = frame[:FRAME_LEN]
        # Apply hann window
        x[i, :FRAME_LEN] = x[i, :FRAME_LEN] * window
    
    # Zero-pad for FFT
    x_padded = np.zeros((batch, n_fft))
    x_padded[:, :FRAME_LEN] = x
    
    # FFT
    spec = np.fft.rfft(x_padded, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    
    # Mel filterbank
    mel_fb = build_mel_filterbank(sample_rate, n_fft, n_mels)
    mel = np.dot(power, mel_fb.T)
    
    # Log
    log_mel = np.log(np.clip(mel, a_min=1e-9, a_max=None))
    
    if single:
        return log_mel[0]
    return log_mel


def extract_context_features(audio_segment, sample_rate=SAMPLE_RATE, n_fft=512, n_mels=N_MELS, n_frames=N_FRAMES):
    """
    Extract N_FRAMES consecutive mel features from audio and concatenate.
    Returns flattened feature vector (n_mels * n_frames,).
    
    Args:
        audio_segment: (n_frames * frame_len,) audio array
        
    Returns:
        (n_mels * n_frames,) concatenated features
    """
    # Ensure audio is exact length
    expected_len = FRAME_LEN * n_frames
    if len(audio_segment) < expected_len:
        audio_segment = np.pad(audio_segment, (0, expected_len - len(audio_segment)))
    else:
        audio_segment = audio_segment[:expected_len]
    
    # Extract features for each frame
    features_list = []
    for frame_idx in range(n_frames):
        start = frame_idx * FRAME_LEN
        end = start + FRAME_LEN
        frame = audio_segment[start:end]
        
        # Extract mel features for this frame
        frame_mel = extract_log_mel_features(frame, sample_rate, n_fft, n_mels)
        features_list.append(frame_mel)
    
    # Concatenate all frames
    context_features = np.concatenate(features_list)  # (n_mels * n_frames,)
    return context_features


def load_audio(filepath, sr=SAMPLE_RATE):
    """Load audio with automatic resampling."""
    try:
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        return audio
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return None


def mix_speech_noise(speech, noise, snr_db=10):
    """
    Mix speech and noise at given SNR.
    SNR = 10 * log10(P_speech / P_noise)
    """
    # Ensure same length
    min_len = min(len(speech), len(noise))
    speech = speech[:min_len]
    noise = noise[:min_len]
    
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Scale noise to achieve target SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(speech_power / (snr_linear * noise_power + 1e-9))
    
    mixed = speech + noise_scale * noise
    return mixed


# ──────────────────────────────────────────────────────────────────────────────
# 2. Data Inventory (find all speech and noise files)
# ──────────────────────────────────────────────────────────────────────────────

def collect_speech_files(librispeech_dir, max_files=None):
    """Collect all .flac files from LibriSpeech."""
    speech_files = []
    for root, dirs, files in os.walk(librispeech_dir):
        for f in files:
            if f.endswith(".flac"):
                speech_files.append(os.path.join(root, f))
                if max_files and len(speech_files) >= max_files:
                    return speech_files
    return speech_files


def collect_noise_files(musan_noise_dir, max_files=None):
    """Collect all .wav files from MUSAN noise directory."""
    noise_files = []
    for root, dirs, files in os.walk(musan_noise_dir):
        for f in files:
            if f.endswith(".wav"):
                noise_files.append(os.path.join(root, f))
                if max_files and len(noise_files) >= max_files:
                    return noise_files
    return noise_files


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dataset Building
# ──────────────────────────────────────────────────────────────────────────────

def build_real_vad_dataset(split='train', n_samples=5000, snr_range=(5, 15)):
    """
    Build VAD training set from real audio.
    
    Args:
        split: 'train' or 'val'
        n_samples: number of samples to generate
        snr_range: (min_snr, max_snr) for speech+noise samples
    
    Returns:
        (X, y) tensors where:
        - X: (n_samples, FEATURE_DIM)
        - y: (n_samples,) binary labels
    """
    print(f"\n{'━' * 70}")
    print(f"Building {split.upper()} dataset ({n_samples} samples)")
    print(f"{'━' * 70}")
    
    # Select LibriSpeech split
    if split == 'train':
        speech_dir = LIBRISPEECH_TRAIN
    else:
        speech_dir = LIBRISPEECH_DEV
    
    # Collect files
    print("\n[1/4] Collecting audio files...")
    speech_files = collect_speech_files(speech_dir)
    noise_files = collect_noise_files(MUSAN_NOISE_DIR)
    
    print(f"  ✓ Found {len(speech_files)} speech files")
    print(f"  ✓ Found {len(noise_files)} noise files")
    
    if len(speech_files) < 10:
        raise RuntimeError(f"Not enough speech files found in {speech_dir}")
    if len(noise_files) < 10:
        raise RuntimeError(f"Not enough noise files found in {MUSAN_NOISE_DIR}")
    
    # Pre-load some audio and noise for efficiency
    print("\n[2/4] Pre-loading audio files...")
    random.seed(42)
    
    # Load speech segments
    speech_segments = []
    sample_speech_files = random.sample(speech_files, min(50, len(speech_files)))
    for fpath in tqdm(sample_speech_files, desc="Speech"):
        audio = load_audio(fpath)
        if audio is not None and len(audio) > FRAME_LEN * N_FRAMES:
            # Split into fixed-length segments
            for start in range(0, len(audio) - FRAME_LEN * N_FRAMES, FRAME_LEN * N_FRAMES):
                segment = audio[start:start + FRAME_LEN * N_FRAMES]
                if len(segment) == FRAME_LEN * N_FRAMES:
                    speech_segments.append(segment)
    
    print(f"  ✓ Extracted {len(speech_segments)} speech segments")
    
    # Load noise segments
    noise_segments = []
    sample_noise_files = random.sample(noise_files, min(50, len(noise_files)))
    for fpath in tqdm(sample_noise_files, desc="Noise"):
        audio = load_audio(fpath)
        if audio is not None and len(audio) > FRAME_LEN * N_FRAMES:
            # Split into segments
            for start in range(0, len(audio) - FRAME_LEN * N_FRAMES, FRAME_LEN * N_FRAMES):
                segment = audio[start:start + FRAME_LEN * N_FRAMES]
                if len(segment) == FRAME_LEN * N_FRAMES:
                    noise_segments.append(segment)
    
    print(f"  ✓ Extracted {len(noise_segments)} noise segments")
    
    # Generate training samples
    print("\n[3/4] Generating training samples...")
    X_list = []
    y_list = []
    
    for i in tqdm(range(n_samples), desc="Samples"):
        # Decide on sample type (1/3 each: speech only, noise only, mixed)
        sample_type = i % 3
        
        if sample_type == 0:
            # Speech only (label=1)
            speech = random.choice(speech_segments)
            features = extract_context_features(speech, SAMPLE_RATE, 512, N_MELS, N_FRAMES)
            X_list.append(features)
            y_list.append(1)
            
        elif sample_type == 1:
            # Noise only (label=0)
            noise = random.choice(noise_segments)
            features = extract_context_features(noise, SAMPLE_RATE, 512, N_MELS, N_FRAMES)
            X_list.append(features)
            y_list.append(0)
            
        else:
            # Mixed: speech + noise (label=1)
            speech = random.choice(speech_segments)
            noise = random.choice(noise_segments)
            snr = random.uniform(snr_range[0], snr_range[1])
            
            mixed = mix_speech_noise(speech, noise, snr)
            features = extract_context_features(mixed, SAMPLE_RATE, 512, N_MELS, N_FRAMES)
            X_list.append(features)
            y_list.append(1)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"  ✓ Generated X: {X.shape}, y: {y.shape}")
    
    # Convert to torch
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).float()
    
    # Save to disk
    print("\n[4/4] Saving to disk...")
    save_path_x = DATA_PROCESSED_DIR / f"{split}_X.pt"
    save_path_y = DATA_PROCESSED_DIR / f"{split}_y.pt"
    torch.save(X_torch, save_path_x)
    torch.save(y_torch, save_path_y)
    print(f"  ✓ Saved {save_path_x}")
    print(f"  ✓ Saved {save_path_y}")
    
    return X_torch, y_torch


# ──────────────────────────────────────────────────────────────────────────────
# 4. Dataset Class (matching SyntheticVADDataset interface)
# ──────────────────────────────────────────────────────────────────────────────

class RealVADDataset(Dataset):
    """
    Real audio VAD dataset (drop-in replacement for SyntheticVADDataset).
    Loads pre-extracted features from disk.
    """
    def __init__(self, split='train'):
        self.split = split
        
        # Load from disk
        X_path = DATA_PROCESSED_DIR / f"{split}_X.pt"
        y_path = DATA_PROCESSED_DIR / f"{split}_y.pt"
        
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Dataset not found. Run build_dataset() first.")
        
        self.X = torch.load(X_path)
        self.y = torch.load(y_path)
        
        print(f"[{split.upper()}] Loaded X: {self.X.shape}, y: {self.y.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main: Build datasets
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("BUILDING REAL AUDIO DATASETS")
    print("=" * 70)
    
    # Check if raw data exists
    if not LIBRISPEECH_TRAIN.exists():
        raise RuntimeError(f"LibriSpeech not found at {LIBRISPEECH_TRAIN}")
    if not MUSAN_NOISE_DIR.exists():
        raise RuntimeError(f"MUSAN noise not found at {MUSAN_NOISE_DIR}")
    
    # Build train and validation splits
    print("\nBuilding TRAIN set...")
    X_train, y_train = build_real_vad_dataset('train', n_samples=5000, snr_range=(5, 15))
    
    print("\n" + "─" * 70)
    print("\nBuilding VALIDATION set...")
    X_val, y_val = build_real_vad_dataset('val', n_samples=1000, snr_range=(5, 15))
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Train set: X{tuple(X_train.shape)}, y{tuple(y_train.shape)}")
    print(f"  - Speech+noise: {(y_train == 1).sum().item()} samples")
    print(f"  - Noise only: {(y_train == 0).sum().item()} samples")
    print(f"\nVal set: X{tuple(X_val.shape)}, y{tuple(y_val.shape)}")
    print(f"  - Speech+noise: {(y_val == 1).sum().item()} samples")
    print(f"  - Noise only: {(y_val == 0).sum().item()} samples")
    print(f"\nFeature dimension: {X_train.shape[1]} ({N_MELS} mels × {N_FRAMES} frames)")
    print(f"\nDatasets saved to: {DATA_PROCESSED_DIR}")
    print("=" * 70)
    print("\n✓ Ready to train! Update vad_dl_demo.py to use RealVADDataset")


if __name__ == "__main__":
    main()
