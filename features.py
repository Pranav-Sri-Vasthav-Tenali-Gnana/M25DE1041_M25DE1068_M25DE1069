import numpy as np
import librosa
from config import SR, N_MFCC, N_MELS, MAX_FRAMES


def load_and_preprocess(path, sr=SR):
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.effects.preemphasis(y)
    return y


def extract_mfcc_features(path, sr=SR, n_mfcc=N_MFCC):
    y = load_and_preprocess(path, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat


def extract_melspec_image(path, sr=SR, n_mels=N_MELS, max_frames=MAX_FRAMES):
    y = load_and_preprocess(path, sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_db.shape[1] < max_frames:
        pad = max_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    else:
        mel_db = mel_db[:, :max_frames]
    return mel_db


def extract_melspec_normalised(path, mean, std, sr=SR, n_mels=N_MELS, max_frames=MAX_FRAMES):
    y = load_and_preprocess(path, sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_db = (mel_db - mean) / (std + 1e-8)
    if mel_db.shape[1] < max_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_frames - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :max_frames]
    return mel_db


def build_feature_matrices(dataset):
    from tqdm import tqdm
    X_mfcc, X_mel, y_labels = [], [], []
    for path, label in tqdm(dataset, desc="Extracting"):
        try:
            X_mfcc.append(extract_mfcc_features(path))
            X_mel.append(extract_melspec_image(path))
            y_labels.append(label)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    X_mfcc = np.array(X_mfcc, dtype=np.float32)
    X_mel = np.array(X_mel, dtype=np.float32)
    y_array = np.array(y_labels, dtype=np.int32)
    return X_mfcc, X_mel, y_array
