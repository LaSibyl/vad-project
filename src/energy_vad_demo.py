import numpy as np
import matplotlib.pyplot as plt
import librosa
import os


def frame_signal(signal, frame_length, hop_length):
    frames = []
    for start in range(0, len(signal) - frame_length + 1, hop_length):
        frames.append(signal[start:start + frame_length])
    return np.array(frames)


def compute_log_energy(frames):
    energy = np.sum(frames ** 2, axis=1)
    return np.log(energy + 1e-10)


def main():
    audio_path = "data/sample.wav"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # load audio
    signal, sr = librosa.load(audio_path, sr=16000)
    print(f"Duration: {len(signal)/sr:.2f} sec")

    # frame params
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    frames = frame_signal(signal, frame_length, hop_length)
    log_energy = compute_log_energy(frames)

    # threshold
    threshold = np.mean(log_energy)
    pred = (log_energy > threshold).astype(int)

    # time axis
    frame_times = np.arange(len(log_energy)) * hop_length / sr
    audio_times = np.arange(len(signal)) / sr

    # plot
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(audio_times, signal)
    plt.title("Waveform")

    plt.subplot(3, 1, 2)
    plt.plot(frame_times, log_energy)
    plt.axhline(y=threshold, linestyle="--")
    plt.title("Log Energy")

    plt.subplot(3, 1, 3)
    plt.step(frame_times, pred, where="post")
    plt.title("VAD Prediction")

    plt.tight_layout()
    plt.savefig("outputs/vad_result.png")
    plt.show()


if __name__ == "__main__":
    main()