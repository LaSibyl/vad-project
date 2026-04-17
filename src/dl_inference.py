import torch
import numpy as np

def run_dl_inference(model, data_loader):
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            preds = model(X_batch).squeeze()
            predicted = (preds > 0.5).float()

            preds_all.extend(predicted.cpu().numpy())
            labels_all.extend(y_batch.cpu().numpy())

    return np.array(labels_all), np.array(preds_all)