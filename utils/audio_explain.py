import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(
        S_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        ax=ax
    )
    ax.set_title("Audio Mel Spectrogram (Explainability)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    return fig