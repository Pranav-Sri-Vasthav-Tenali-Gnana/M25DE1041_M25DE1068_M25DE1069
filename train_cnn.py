import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config import NUM_CLASSES


def augment_batch(X):
    X = X.copy()
    for i in range(len(X)):
        f0 = np.random.randint(0, 50)
        fw = np.random.randint(0, 15)
        X[i, f0:f0 + fw, :, :] = 0
        t0 = np.random.randint(0, 50)
        tw = np.random.randint(0, 15)
        X[i, :, t0:t0 + tw, :] = 0
    return X


def build_cnn(input_shape=(64, 64, 1), num_classes=NUM_CLASSES):
    inp = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inp, out, name="CNN_ASDR")


def train_cnn(X_mel_train, y_train_oh, X_mel_val, y_val_oh, X_mel_test, y_test_oh):
    mean = X_mel_train.mean()
    std = X_mel_train.std() + 1e-8
    X_tr_norm = (X_mel_train - mean) / std
    X_val_norm = (X_mel_val - mean) / std
    X_te_norm = (X_mel_test - mean) / std

    X_aug = np.concatenate([X_tr_norm] + [augment_batch(X_tr_norm) for _ in range(3)])
    y_aug = np.tile(y_train_oh, (4, 1))
    idx = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]

    keras.backend.clear_session()
    tf.random.set_seed(42)

    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    ]

    t0 = time.time()
    history = model.fit(
        X_aug, y_aug,
        validation_data=(X_val_norm, y_val_oh),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - t0

    loss, acc = model.evaluate(X_te_norm, y_test_oh, verbose=0)
    y_pred = np.argmax(model.predict(X_te_norm, verbose=0), axis=1)
    print(f"\nCNN trained in {elapsed:.1f}s | Test Accuracy: {acc*100:.2f}%")

    return model, history, y_pred, acc, elapsed, mean, std, X_te_norm
