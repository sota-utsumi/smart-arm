"""
Shared feature extraction and OpenCV SVM utilities for weed detection.
"""

from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "images" / "train"
SOURCE_DIR = BASE_DIR / "images" / "source"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "weed_classifier.yml"
SCALER_PATH = MODEL_DIR / "weed_scaler.yml"

IMAGE_SIZE = (128, 128)
HSV_LOWER = np.array([25, 40, 40], dtype=np.uint8)
HSV_UPPER = np.array([95, 255, 255], dtype=np.uint8)
MIN_CONTOUR_AREA = 500
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

LABEL_NOT_WEED = 0
LABEL_WEED = 1
LABEL_MAP = {
    LABEL_NOT_WEED: "not_weed",
    LABEL_WEED: "weed",
}


def extract_features(image_bgr):
    """Extract the fixed-length feature vector used by both Python and C++."""
    resized = cv2.resize(image_bgr, IMAGE_SIZE)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    hist_features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.append(hist)
    color_feat = np.concatenate(hist_features)

    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    green_ratio = np.count_nonzero(mask) / mask.size

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    texture_feat = np.array(
        [
            magnitude.mean(),
            magnitude.std(),
            magnitude.max(),
            np.count_nonzero(magnitude > magnitude.mean()) / magnitude.size,
        ],
        dtype=np.float32,
    )

    feature_row = np.concatenate([color_feat, [green_ratio], texture_feat]).astype(
        np.float32
    )
    return feature_row.reshape(1, -1)


def get_image_paths(directory):
    """Return image files in a directory."""
    if not directory.exists():
        return []

    paths = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def compute_scaler(samples):
    """Compute feature-wise mean and standard deviation."""
    feature_mean = samples.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = samples.std(axis=0, keepdims=True).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    return feature_mean, feature_std


def normalize_features(samples, feature_mean, feature_std):
    """Apply the training-time scaler to a feature matrix."""
    return ((samples - feature_mean) / feature_std).astype(np.float32)


def save_scaler(path, feature_mean, feature_std):
    """Save scaler statistics in the same format as OpenCV C++."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise RuntimeError(f"failed to open scaler file for writing: {path}")

    storage.write("feature_mean", feature_mean)
    storage.write("feature_std", feature_std)
    storage.release()


def load_scaler(path):
    """Load scaler statistics from an OpenCV FileStorage file."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"failed to open scaler file for reading: {path}")

    feature_mean = storage.getNode("feature_mean").mat()
    feature_std = storage.getNode("feature_std").mat()
    storage.release()

    if feature_mean is None or feature_std is None:
        raise RuntimeError(f"invalid scaler data: {path}")

    return feature_mean.astype(np.float32), feature_std.astype(np.float32)


class WeedClassifier:
    """OpenCV SVM wrapper shared by learn.py and weed_detector.py."""

    def __init__(self):
        self._svm = None
        self.feature_mean = None
        self.feature_std = None

    def train(self, samples, labels):
        """Fit the classifier and scaler from feature rows."""
        if not samples or len(samples) != len(labels):
            raise RuntimeError("training data is empty or inconsistent")

        training_data = np.vstack(samples).astype(np.float32)
        label_data = np.asarray(labels, dtype=np.int32).reshape(-1, 1)

        self.feature_mean, self.feature_std = compute_scaler(training_data)
        normalized = normalize_features(
            training_data,
            self.feature_mean,
            self.feature_std,
        )

        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setC(10.0)
        svm.setGamma(1.0 / normalized.shape[1])
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
        svm.train(normalized, cv2.ml.ROW_SAMPLE, label_data)
        self._svm = svm

    def save(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """Persist the model and scaler to disk."""
        if self._svm is None:
            raise RuntimeError("classifier is not trained")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        self._svm.save(str(model_path))
        save_scaler(scaler_path, self.feature_mean, self.feature_std)

    def load(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """Load the model and scaler from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"scaler file not found: {scaler_path}")

        self._svm = cv2.ml.SVM_load(str(model_path))
        self.feature_mean, self.feature_std = load_scaler(scaler_path)
        return self

    def predict(self, sample):
        """Predict the class label for a single feature row."""
        if self._svm is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("classifier is not loaded")

        normalized = normalize_features(
            np.asarray(sample, dtype=np.float32).reshape(1, -1),
            self.feature_mean,
            self.feature_std,
        )
        _, results = self._svm.predict(normalized)
        return int(results[0, 0])

    def predict_with_confidence(self, sample):
        """Predict the class label and a margin-derived confidence score."""
        if self._svm is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("classifier is not loaded")

        normalized = normalize_features(
            np.asarray(sample, dtype=np.float32).reshape(1, -1),
            self.feature_mean,
            self.feature_std,
        )
        _, results = self._svm.predict(normalized)
        _, raw_output = self._svm.predict(
            normalized,
            flags=cv2.ml.STAT_MODEL_RAW_OUTPUT,
        )
        margin = abs(float(raw_output[0, 0]))
        confidence = 1.0 / (1.0 + np.exp(-margin))
        return int(results[0, 0]), confidence
