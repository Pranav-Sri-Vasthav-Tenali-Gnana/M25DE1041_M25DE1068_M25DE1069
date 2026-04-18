import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, model_name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n{'─'*45}")
    print(f" Model     : {model_name}")
    print(f" Accuracy  : {acc*100:.2f}%")
    print(f" Precision : {prec*100:.2f}%  (macro)")
    print(f" Recall    : {rec*100:.2f}%  (macro)")
    print(f" F1-Score  : {f1*100:.2f}%  (macro)")
    print(f"{'─'*45}")
    return dict(model=model_name, accuracy=acc, precision=prec, recall=rec, f1=f1)


def plot_confusion_matrices(y_test, y_pred_svm, y_pred_rf, y_pred_cnn):
    digit_labels = [str(d) for d in range(10)]
    models_preds = [
        ("SVM",           y_pred_svm, "Blues"),
        ("Random Forest", y_pred_rf,  "Greens"),
        ("CNN",           y_pred_cnn, "Purples"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight='bold')
    for ax, (name, preds, cmap) in zip(axes, models_preds):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=digit_labels, yticklabels=digit_labels,
                    linewidths=0.3, cbar=False)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()


def print_classification_reports(y_test, y_pred_svm, y_pred_rf, y_pred_cnn):
    digit_labels = [str(d) for d in range(10)]
    for name, preds in [("SVM", y_pred_svm), ("Random Forest", y_pred_rf), ("CNN", y_pred_cnn)]:
        print(f"\n{'═'*55}")
        print(f"  Classification Report — {name}")
        print('═' * 55)
        print(classification_report(y_test, preds, target_names=digit_labels))


def plot_model_comparison(svm_metrics, rf_metrics, cnn_metrics):
    metrics_list = [svm_metrics, rf_metrics, cnn_metrics]
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']
    colors = ['#4C72B0', '#55A868', '#C44E52']
    x = np.arange(len(metric_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, clr) in enumerate(zip(metrics_list, colors)):
        vals = [m[k] * 100 for k in metric_names]
        bars = ax.bar(x + i * w, vals, w, label=m['model'], color=clr, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title("Model Comparison — SVM vs Random Forest vs CNN", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()


def plot_accuracy_vs_time(svm_accuracy, svm_time, rf_accuracy, rf_time, cnn_accuracy, cnn_time):
    fig, ax = plt.subplots(figsize=(7, 5))
    model_data = [
        ("SVM",           svm_accuracy * 100, svm_time,  '#4C72B0'),
        ("Random Forest", rf_accuracy * 100,  rf_time,   '#55A868'),
        ("CNN",           cnn_accuracy * 100, cnn_time,  '#C44E52'),
    ]
    for name, acc, t, clr in model_data:
        ax.scatter(t, acc, s=200, color=clr, label=name, zorder=5)
        ax.annotate(name, (t, acc), textcoords='offset points', xytext=(8, 4), fontsize=11)
    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Computational Cost", fontsize=13, fontweight='bold')
    ax.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("accuracy_vs_time.png", dpi=150)
    plt.show()
