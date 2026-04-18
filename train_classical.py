import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def train_svm(X_train, y_train, X_test, y_test):
    print("Training SVM...")
    t0 = time.time()
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  SVM trained in {elapsed:.2f}s | Test Accuracy: {acc*100:.2f}%")
    return pipeline, y_pred, acc, elapsed


def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training Random Forest...")
    t0 = time.time()
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Random Forest trained in {elapsed:.2f}s | Test Accuracy: {acc*100:.2f}%")
    return pipeline, y_pred, acc, elapsed
