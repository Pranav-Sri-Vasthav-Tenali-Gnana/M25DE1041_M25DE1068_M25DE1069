# Automatic Spoken Digit Recognition (ASDR)

**Course:** Speech Understanding  
**Team Members:**
- Abhinav Tote (M25DE1041)
- Aakanksha Nalamati (M25DE1068)
- Pranav Sri Vasthav Tenali Gnana (M25DE1069)

---

## Overview

This project implements a complete pipeline for Automatic Spoken Digit Recognition comparing:
- **Classical ML:** SVM (RBF kernel), Random Forest — trained on MFCC features
- **Deep Learning:** CNN — trained on Mel-Spectrogram images

Dataset: [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset)

---

## Project Structure

```
spoken_digit_recognition/
├── config.py           # Constants and hyperparameters
├── data_loader.py      # FSDD download and dataset loading
├── features.py         # MFCC and Mel-Spectrogram extraction
├── train_classical.py  # SVM and Random Forest training
├── train_cnn.py        # CNN architecture, augmentation, training
├── evaluate.py         # Metrics, confusion matrices, comparison plots
├── robustness.py       # Noise robustness evaluation (AWGN at varying SNR)
├── fairness.py         # Per-speaker accuracy fairness evaluation
├── visualize.py        # EDA, waveform, feature, and CNN activation plots
├── main.py             # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/spoken-digit-recognition.git
cd spoken-digit-recognition
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow requires Python 3.8–3.11. If you have a GPU, install `tensorflow[and-cuda]` instead.

---

## Running the Project

```bash
python main.py
```

This will:
1. Download the FSDD dataset (~900 audio files) into `fsdd_recordings/`
2. Run EDA and save visualisation plots
3. Extract MFCC and Mel-Spectrogram features
4. Train SVM, Random Forest, and CNN models
5. Evaluate and compare all models
6. Run robustness (noisy audio) and fairness (per-speaker) analyses
7. Save all plots as PNG files in the project directory

---

## Output Files

| File | Description |
|------|-------------|
| `class_distribution.png` | Sample count per digit |
| `waveforms.png` | Waveform for each digit |
| `features_viz.png` | MFCC & Mel-Spectrogram visualisation |
| `cnn_training_curves.png` | Loss and accuracy curves |
| `confusion_matrices.png` | Per-model confusion matrix |
| `model_comparison.png` | Grouped bar chart of metrics |
| `accuracy_vs_time.png` | Accuracy vs training time scatter |
| `robustness_noise.png` | Accuracy under varying SNR |
| `per_speaker_accuracy.png` | Per-speaker fairness analysis |
| `cnn_activation_maps.png` | CNN conv layer activation maps |

---

## System Workflow

```
Audio Input (0–9 spoken digits)
        ↓
Preprocessing (noise removal, normalization, pre-emphasis)
        ↓
Feature Extraction
   ├── MFCC (mean + std) → Classical ML
   └── Mel-Spectrogram   → CNN
        ↓
Model Training
   ├── SVM (RBF kernel)
   ├── Random Forest
   └── CNN (with SpecAugment)
        ↓
Evaluation (Accuracy, Precision, Recall, F1, Robustness, Fairness)
```

---

## References

1. Sen et al., CNN based spoken digit recognition, 2021.
2. Rakshith et al., MFCC based spoken digit system, 2021.
3. Nasr-Esfahani et al., CNN-BiGRU spoken digit recognition, 2024.
4. Jakobovski, Free Spoken Digit Dataset (FSDD), GitHub.
5. McFee et al., librosa: Audio and Music Signal Analysis in Python, 2015.
