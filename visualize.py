import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tensorflow import keras
from config import SR


def plot_class_distribution(label_counts):
    fig, ax = plt.subplots(figsize=(9, 4))
    digits_sorted = sorted(label_counts.keys())
    counts_sorted = [label_counts[d] for d in digits_sorted]
    bars = ax.bar([str(d) for d in digits_sorted], counts_sorted,
                  color=plt.cm.tab10(np.linspace(0, 1, 10)), edgecolor='white')
    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution — FSDD", fontsize=14, fontweight='bold')
    for bar, cnt in zip(bars, counts_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()


def plot_waveforms(dataset, sr=SR):
    sample_files = {}
    for path, label in dataset:
        if label not in sample_files:
            sample_files[label] = path
        if len(sample_files) == 10:
            break

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    fig.suptitle("Waveforms for Digits 0–9", fontsize=15, fontweight='bold')
    for digit in range(10):
        ax = axes[digit // 5][digit % 5]
        y, sr_loaded = librosa.load(sample_files[digit], sr=sr)
        times = np.linspace(0, len(y) / sr_loaded, num=len(y))
        ax.plot(times, y, linewidth=0.6, color=plt.cm.tab10(digit / 10))
        ax.set_title(f"Digit '{digit}'", fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig("waveforms.png", dpi=150)
    plt.show()
    return sample_files


def plot_feature_representations(sample_path, sr=SR):
    y, _ = librosa.load(sample_path, sr=sr)
    y_preemph = librosa.effects.preemphasis(y)
    mfcc = librosa.feature.mfcc(y=y_preemph, sr=sr, n_mfcc=40)
    mel = librosa.feature.melspectrogram(y=y_preemph, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Feature Representations for digit '5'", fontsize=13, fontweight='bold')

    axes[0].plot(np.linspace(0, len(y) / sr, len(y)), y, color='steelblue', linewidth=0.8)
    axes[0].set_title("Waveform (after pre-emphasis)")
    axes[0].set_xlabel("Time (s)")

    img1 = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1], cmap='coolwarm')
    fig.colorbar(img1, ax=axes[1])
    axes[1].set_title("MFCC (40 coefficients)")

    img2 = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2], cmap='magma')
    fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')
    axes[2].set_title("Mel-Spectrogram (64 mels)")

    plt.tight_layout()
    plt.savefig("features_viz.png", dpi=150)
    plt.show()


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs_ran = range(1, len(history.history['loss']) + 1)

    axes[0].plot(epochs_ran, history.history['loss'], label='Train Loss', color='royalblue')
    axes[0].plot(epochs_ran, history.history['val_loss'], label='Val Loss', color='tomato', linestyle='--')
    axes[0].set_title('Loss Curve', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs_ran, history.history['accuracy'], label='Train Acc', color='royalblue')
    axes[1].plot(epochs_ran, history.history['val_accuracy'], label='Val Acc', color='tomato', linestyle='--')
    axes[1].set_title('Accuracy Curve', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cnn_training_curves.png", dpi=150)
    plt.show()


def plot_cnn_activation_maps(cnn_model, X_mel_test_cnn, y_test):
    conv_layer_name = [l.name for l in cnn_model.layers if 'conv2d' in l.name][-1]
    feat_model = keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer(conv_layer_name).output
    )

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    fig.suptitle("Mean Conv Feature Map per Digit — CNN Interpretability",
                 fontsize=13, fontweight='bold')

    for digit in range(10):
        ax = axes[digit // 5][digit % 5]
        idx = np.where(y_test == digit)[0]
        if len(idx) == 0:
            continue
        samples = X_mel_test_cnn[idx]
        activations = feat_model.predict(samples, verbose=0)
        mean_act = activations.mean(axis=(0, 3))
        ax.imshow(mean_act, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Digit '{digit}'", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("cnn_activation_maps.png", dpi=150)
    plt.show()
