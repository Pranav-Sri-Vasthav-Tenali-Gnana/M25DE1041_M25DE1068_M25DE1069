import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.metrics import accuracy_score
from features import load_and_preprocess
from config import SR, N_MFCC, N_MELS, MAX_FRAMES


def add_awgn_noise(signal, snr_db):
    sig_power = np.mean(signal ** 2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)
    return signal + noise


def noisy_mfcc(path, snr_db, sr=SR, n_mfcc=N_MFCC):
    y = load_and_preprocess(path, sr)
    y = add_awgn_noise(y, snr_db)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def noisy_mel_normalised(path, snr_db, mean, std, sr=SR, n_mels=N_MELS, max_frames=MAX_FRAMES):
    y = load_and_preprocess(path, sr)
    y = add_awgn_noise(y, snr_db)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_db = (mel_db - mean) / (std + 1e-8)
    if mel_db.shape[1] < max_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_frames - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :max_frames]
    return mel_db


def evaluate_robustness(dataset, svm_pipeline, rf_pipeline, cnn_model, mean, std):
    snr_levels = [30, 20, 10, 5, 0]

    test_paths_list, test_labels_list = [], []
    cnt_per_class = {d: 0 for d in range(10)}
    for p, lbl in dataset:
        if cnt_per_class[lbl] < 5:
            test_paths_list.append(p)
            test_labels_list.append(lbl)
            cnt_per_class[lbl] += 1

    results = {"SVM": [], "Random Forest": [], "CNN": []}

    for snr in snr_levels:
        X_noisy_mfcc, X_noisy_mel = [], []
        for p in test_paths_list:
            X_noisy_mfcc.append(noisy_mfcc(p, snr))
            X_noisy_mel.append(noisy_mel_normalised(p, snr, mean, std))
        X_noisy_mfcc = np.array(X_noisy_mfcc, dtype=np.float32)
        X_noisy_mel = np.array(X_noisy_mel, dtype=np.float32)[..., np.newaxis]
        y_noise_test = np.array(test_labels_list)

        acc_svm = accuracy_score(y_noise_test, svm_pipeline.predict(X_noisy_mfcc))
        acc_rf  = accuracy_score(y_noise_test, rf_pipeline.predict(X_noisy_mfcc))
        acc_cnn = accuracy_score(y_noise_test,
                                 np.argmax(cnn_model.predict(X_noisy_mel, verbose=0), axis=1))

        results["SVM"].append(acc_svm * 100)
        results["Random Forest"].append(acc_rf * 100)
        results["CNN"].append(acc_cnn * 100)
        print(f"  SNR {snr:>2}dB → SVM: {acc_svm*100:.1f}%  RF: {acc_rf*100:.1f}%  CNN: {acc_cnn*100:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, clr in [("SVM", '#4C72B0'), ("Random Forest", '#55A868'), ("CNN", '#C44E52')]:
        ax.plot(snr_levels, results[name], marker='o', label=name, color=clr, linewidth=2)
    ax.set_xlabel("SNR (dB)  ← more noise", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Robustness to Additive Noise", fontsize=13, fontweight='bold')
    ax.set_xticks(snr_levels)
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("robustness_noise.png", dpi=150)
    plt.show()

    return results
