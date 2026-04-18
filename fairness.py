import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score
from features import extract_mfcc_features, extract_melspec_normalised
from config import SPEAKERS


def evaluate_per_speaker(dataset, svm_pipeline, rf_pipeline, cnn_model, mean, std):
    speaker_data = {s: {'paths': [], 'labels': []} for s in SPEAKERS}
    for path, label in dataset:
        fname = Path(path).stem
        parts = fname.split('_')
        if len(parts) >= 2:
            spk = '_'.join(parts[1:-1])
            if spk in speaker_data:
                speaker_data[spk]['paths'].append(path)
                speaker_data[spk]['labels'].append(label)

    speaker_accs = {'SVM': {}, 'RF': {}, 'CNN': {}}

    for spk, data in speaker_data.items():
        if len(data['paths']) < 5:
            continue
        X_spk_mfcc, X_spk_mel, y_spk = [], [], []
        for p, lbl in zip(data['paths'], data['labels']):
            try:
                X_spk_mfcc.append(extract_mfcc_features(p))
                X_spk_mel.append(extract_melspec_normalised(p, mean, std))
                y_spk.append(lbl)
            except Exception:
                pass
        if not y_spk:
            continue
        X_spk_mfcc = np.array(X_spk_mfcc, dtype=np.float32)
        X_spk_mel = np.array(X_spk_mel, dtype=np.float32)[..., np.newaxis]
        y_spk = np.array(y_spk)

        speaker_accs['SVM'][spk] = accuracy_score(y_spk, svm_pipeline.predict(X_spk_mfcc)) * 100
        speaker_accs['RF'][spk]  = accuracy_score(y_spk, rf_pipeline.predict(X_spk_mfcc)) * 100
        speaker_accs['CNN'][spk] = accuracy_score(
            y_spk, np.argmax(cnn_model.predict(X_spk_mel, verbose=0), axis=1)) * 100

    spk_names = list(speaker_accs['SVM'].keys())
    x = np.arange(len(spk_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, clr) in enumerate([('SVM', '#4C72B0'), ('RF', '#55A868'), ('CNN', '#C44E52')]):
        vals = [speaker_accs[name].get(s, 0) for s in spk_names]
        ax.bar(x + i * w, vals, w, label=name, color=clr, edgecolor='white')
    ax.set_xticks(x + w)
    ax.set_xticklabels(spk_names, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Per-Speaker Accuracy (Fairness Evaluation)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("per_speaker_accuracy.png", dpi=150)
    plt.show()

    print("\nPer-speaker accuracy summary:")
    for spk in spk_names:
        print(f"  {spk:<12} | SVM: {speaker_accs['SVM'].get(spk, 0):.1f}%  "
              f"RF: {speaker_accs['RF'].get(spk, 0):.1f}%  "
              f"CNN: {speaker_accs['CNN'].get(spk, 0):.1f}%")

    return speaker_accs
