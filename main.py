import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
np.random.seed(42)
tf.random.set_seed(42)

from config import NUM_CLASSES
from data_loader import download_fsdd, get_label_counts
from features import build_feature_matrices
from visualize import (
    plot_class_distribution, plot_waveforms,
    plot_feature_representations, plot_training_curves,
    plot_cnn_activation_maps
)
from train_classical import train_svm, train_random_forest
from train_cnn import train_cnn
from evaluate import (
    compute_metrics, plot_confusion_matrices,
    print_classification_reports, plot_model_comparison,
    plot_accuracy_vs_time
)
from robustness import evaluate_robustness
from fairness import evaluate_per_speaker


def main():
    print("=" * 60)
    print("  Automatic Spoken Digit Recognition (ASDR)")
    print("=" * 60)

    dataset = download_fsdd()
    label_counts = get_label_counts(dataset)
    print(f"\nDataset ready — {len(dataset)} files | Classes: {len(label_counts)}")
    print("Samples per digit:", dict(sorted(label_counts.items())))

    plot_class_distribution(label_counts)
    sample_files = plot_waveforms(dataset)
    plot_feature_representations(sample_files[5])

    print("\nExtracting features...")
    X_mfcc, X_mel, y_array = build_feature_matrices(dataset)
    print(f"X_mfcc: {X_mfcc.shape} | X_mel: {X_mel.shape} | y: {y_array.shape}")

    (X_mfcc_trainval, X_mfcc_test,
     X_mel_trainval,  X_mel_test,
     y_trainval,      y_test) = train_test_split(
        X_mfcc, X_mel, y_array,
        test_size=0.15, stratify=y_array, random_state=42
    )
    (X_mfcc_train, X_mfcc_val,
     X_mel_train,  X_mel_val,
     y_train,      y_val) = train_test_split(
        X_mfcc_trainval, X_mel_trainval, y_trainval,
        test_size=0.15 / 0.85, stratify=y_trainval, random_state=42
    )
    print(f"\nSplit → Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   NUM_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

    X_mel_train_cnn = X_mel_train[..., np.newaxis]
    X_mel_val_cnn   = X_mel_val[...,   np.newaxis]
    X_mel_test_cnn  = X_mel_test[...,  np.newaxis]

    svm_pipeline, y_pred_svm, svm_accuracy, svm_time = train_svm(
        X_mfcc_train, y_train, X_mfcc_test, y_test)

    rf_pipeline, y_pred_rf, rf_accuracy, rf_time = train_random_forest(
        X_mfcc_train, y_train, X_mfcc_test, y_test)

    cnn_model, history, y_pred_cnn, cnn_accuracy, cnn_time, mean, std, X_te_norm = train_cnn(
        X_mel_train_cnn, y_train_oh,
        X_mel_val_cnn,   y_val_oh,
        X_mel_test_cnn,  y_test_oh
    )

    plot_training_curves(history)

    svm_metrics = compute_metrics(y_test, y_pred_svm, "SVM (RBF kernel)")
    rf_metrics  = compute_metrics(y_test, y_pred_rf,  "Random Forest")
    cnn_metrics = compute_metrics(y_test, y_pred_cnn, "CNN (Mel-Spectrogram)")

    plot_confusion_matrices(y_test, y_pred_svm, y_pred_rf, y_pred_cnn)
    print_classification_reports(y_test, y_pred_svm, y_pred_rf, y_pred_cnn)
    plot_model_comparison(svm_metrics, rf_metrics, cnn_metrics)
    plot_accuracy_vs_time(svm_accuracy, svm_time, rf_accuracy, rf_time, cnn_accuracy, cnn_time)

    print("\nRunning robustness evaluation (noisy audio)...")
    evaluate_robustness(dataset, svm_pipeline, rf_pipeline, cnn_model, mean, std)

    print("\nRunning per-speaker fairness evaluation...")
    evaluate_per_speaker(dataset, svm_pipeline, rf_pipeline, cnn_model, mean, std)

    print("\nVisualising CNN activation maps...")
    plot_cnn_activation_maps(cnn_model, X_mel_test_cnn, y_test)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {
            'Model':          m['model'],
            'Features':       'MFCC (mean+std)' if 'CNN' not in m['model'] else 'Mel-Spectrogram',
            'Accuracy (%)':   f"{m['accuracy']*100:.2f}",
            'Precision (%)':  f"{m['precision']*100:.2f}",
            'Recall (%)':     f"{m['recall']*100:.2f}",
            'F1-Score (%)':   f"{m['f1']*100:.2f}",
            'Train Time (s)': f"{t:.1f}"
        }
        for m, t in [(svm_metrics, svm_time), (rf_metrics, rf_time), (cnn_metrics, cnn_time)]
    ])
    summary_df.set_index('Model', inplace=True)
    print(summary_df.to_string())
    print("\nAll outputs saved as PNG files in the current directory.")


if __name__ == "__main__":
    main()
