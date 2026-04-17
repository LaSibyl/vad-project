import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from evaluation import compute_metrics, print_metrics
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
    print(f"\nThreshold: {threshold:.4f}")

    metrics_raw = compute_metrics(gt, vad_raw)
    metrics_smooth = compute_metrics(gt, vad_smooth)

    print_metrics("Raw VAD", metrics_raw)
    print_metrics("Smoothed VAD", metrics_smooth)

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