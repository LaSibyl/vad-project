import numpy as np
import torch
from torch.utils.data import DataLoader

from vad_dl_demo import LightweightCNN_VAD, SyntheticVADDataset


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