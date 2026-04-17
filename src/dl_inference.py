import numpy as np
import torch
import librosa
from torch.utils.data import DataLoader

from vad_dl_demo import LightweightCNN_VAD, SyntheticVADDataset, LogMelExtractor


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

            probs = model(X_batch).squeeze()
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


def load_audio_frames(audio_path, frame_len=320, hop_len=160):
    """
    Split audio into frames using librosa.
    Matches Energy VAD framing for consistency:
      frame_len=320 (20ms @ 16kHz)
      hop_len=160 (10ms @ 16kHz, 50% overlap)
    
    Returns:
        frames: (num_frames, frame_len) array
    """
    signal, sr = librosa.load(audio_path, sr=16000)

    # Use librosa for consistent framing (no manual loop)
    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len)
    frames = frames.T  # Transpose to (num_frames, frame_len)
    
    print(f"Frames shape: {frames.shape}")
    return frames


def build_cnn_inputs(frames, n_frames=5, return_stats=False, normalize=False):
    """
    Convert raw frames → log-mel → context window
    
    Args:
        frames: (num_frames, frame_len) audio frames
        n_frames: context window size (default 5)
        return_stats: if True, return feature statistics
        normalize: if True, normalize each window to zero mean and unit variance
    
    Returns:
        If return_stats=False: (T-4, 200) array of CNN inputs
        If return_stats=True:  (inputs, stats_dict)
    """
    extractor = LogMelExtractor()

    features = []
    for frame in frames:
        frame_tensor = torch.tensor(frame).unsqueeze(0)  # (1, 320)
        mel = extractor(frame_tensor).squeeze().numpy()  # (40,)
        features.append(mel)

    features = np.array(features)  # (T, 40)

    # sliding window: 5 frames
    inputs = []
    for i in range(len(features) - n_frames + 1):
        window = features[i:i+n_frames].flatten()  # (40×5=200)
        
        # Normalize: zero mean, unit variance
        if normalize:
            window = (window - window.mean()) / (window.std() + 1e-6)
        
        inputs.append(window)

    inputs = np.array(inputs)  # (T-4, 200)
    
    if return_stats:
        stats = {
            "min": inputs.min(),
            "max": inputs.max(),
            "mean": inputs.mean(),
            "std": inputs.std(),
            "median": np.median(inputs),
            "shape": inputs.shape,
            "data": inputs.flatten()  # for histogram
        }
        return inputs, stats
    
    return inputs


def run_dl_on_audio(audio_path, checkpoint_path, return_feature_stats=False, normalize_inputs=False):
    """
    Run CNN inference on real audio.
    
    Args:
        audio_path: path to audio file
        checkpoint_path: path to saved model checkpoint
        return_feature_stats: if True, return input feature statistics
        normalize_inputs: if True, normalize each input window to zero mean/unit variance
                         (useful for debugging distribution mismatch)
    
    Returns:
        dict with y_prob, y_pred, and optionally feature_stats
    """
    device = torch.device("cpu")

    # 1. load frames
    frames = load_audio_frames(audio_path)

    # 2. build CNN inputs (with optional normalization)
    X, feature_stats = build_cnn_inputs(frames, return_stats=True, normalize=normalize_inputs)
    
    if normalize_inputs:
        print(f"✓ Input normalization enabled (per-window zero mean/unit variance)")

    # 3. load model
    model = LightweightCNN_VAD().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. inference
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        probs = model(X_tensor).squeeze().numpy()
        preds = (probs > 0.5).astype(int)

    result = {
        "y_prob": probs,
        "y_pred": preds
    }
    
    if return_feature_stats:
        result["feature_stats"] = feature_stats
    
    return result

