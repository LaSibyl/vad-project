import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "F1": f1,
        "FAR": far,
        "MISS": miss,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    print(f"F1 Score         : {metrics['F1']:.4f}")
    print(f"False Alarm Rate : {metrics['FAR']:.4f}")
    print(f"Miss Rate        : {metrics['MISS']:.4f}")
    print(f"TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, TP={metrics['TP']}")