from typing import Optional

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(audio: np.ndarray, sample_rate: int, title: str, 
                  figsize: tuple = (12, 3),
                  y_limits=None):
    """
    Plot a time-domain waveform.
    """
    duration = len(audio) / float(sample_rate)
    time_axis = np.linspace(0.0, duration, num=len(audio))

    plt.figure(figsize=figsize)
    plt.plot(time_axis, audio)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    if y_limits is not None:
        plt.ylim(y_limits)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Spectrogram",
    n_fft: int = 1024,
    hop_length: int = 512,
    cmap: str = "inferno",
    figsize: tuple = (10, 4),
) -> None:
    """
    Plot a standard spectrogram using matplotlib.specgram.
    """
    plt.figure(figsize=figsize)
    plt.specgram(audio, Fs=sample_rate, NFFT=n_fft, noverlap=hop_length, cmap=cmap)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.tight_layout()
    plt.show()

