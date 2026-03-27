import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from sklearn.metrics import f1_score, confusion_matrix
from scipy.signal import medfilt


def frame_signal(signal, frame_length, hop_length):
    frames = []
    for start in range(0, len(signal) - frame_length + 1, hop_length):
        frames.append(signal[start:start + frame_length])
    return np.array(frames)


def compute_log_energy(frames):
    energy = np.sum(frames ** 2, axis=1)
    return np.log(energy + 1e-10)


def build_ground_truth(frame_times):
    """
    Manual ground truth for your current sample.
    Adjust these time intervals based on your waveform.
    """
    gt = np.zeros_like(frame_times, dtype=int)

    # Example: speech roughly from 2.6s to 4.15s
    gt[(frame_times >= 2.6) & (frame_times <= 4.15)] = 1

    return gt


def compute_far_miss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return far, miss_rate, tn, fp, fn, tp


def main():
    audio_path = "data/sample.wav"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # load audio
    signal, sr = librosa.load(audio_path, sr=16000)
    print(f"Duration: {len(signal)/sr:.2f} sec")

    # frame params
    frame_length = int(0.025 * sr)   # 25 ms
    hop_length = int(0.010 * sr)     # 10 ms

    frames = frame_signal(signal, frame_length, hop_length)
    log_energy = compute_log_energy(frames)

    # threshold baseline
    threshold = np.mean(log_energy)
    vad_raw = (log_energy > threshold).astype(int)

    # smoothing
    vad_smooth = medfilt(vad_raw, kernel_size=5)

    # time axis
    frame_times = np.arange(len(log_energy)) * hop_length / sr
    audio_times = np.arange(len(signal)) / sr

    # ground truth
    gt = build_ground_truth(frame_times)

    # evaluation
    f1_raw = f1_score(gt, vad_raw)
    far_raw, miss_raw, tn_r, fp_r, fn_r, tp_r = compute_far_miss(gt, vad_raw)

    f1_smooth = f1_score(gt, vad_smooth)
    far_smooth, miss_smooth, tn_s, fp_s, fn_s, tp_s = compute_far_miss(gt, vad_smooth)

    print("\n=== Raw VAD Evaluation ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"F1 Score: {f1_raw:.4f}")
    print(f"False Alarm Rate: {far_raw:.4f}")
    print(f"Miss Rate: {miss_raw:.4f}")
    print(f"TN={tn_r}, FP={fp_r}, FN={fn_r}, TP={tp_r}")

    print("\n=== Smoothed VAD Evaluation ===")
    print(f"F1 Score: {f1_smooth:.4f}")
    print(f"False Alarm Rate: {far_smooth:.4f}")
    print(f"Miss Rate: {miss_smooth:.4f}")
    print(f"TN={tn_s}, FP={fp_s}, FN={fn_s}, TP={tp_s}")

    # plot
    plt.figure(figsize=(12, 10))

    plt.subplot(5, 1, 1)
    plt.plot(audio_times, signal)
    plt.title("Waveform")

    plt.subplot(5, 1, 2)
    plt.plot(frame_times, log_energy)
    plt.axhline(y=threshold, linestyle="--")
    plt.title("Log Energy")

    plt.subplot(5, 1, 3)
    plt.step(frame_times, gt, where="post")
    plt.title("Ground Truth")

    plt.subplot(5, 1, 4)
    plt.step(frame_times, vad_raw, where="post")
    plt.title("Raw VAD Prediction")

    plt.subplot(5, 1, 5)
    plt.step(frame_times, vad_smooth, where="post")
    plt.title("Smoothed VAD Prediction")

    plt.tight_layout()
    plt.savefig("outputs/vad_eval_result.png")
    plt.show()


if __name__ == "__main__":
    main()