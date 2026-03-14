"""
Train the shared weed classifier model used by Python and C++ detectors.
"""

import argparse
import sys
from pathlib import Path

import cv2

from weed_model import (
    LABEL_NOT_WEED,
    LABEL_WEED,
    MODEL_PATH,
    SCALER_PATH,
    TRAIN_DIR,
    WeedClassifier,
    extract_features,
    get_image_paths,
)


def collect_training_data(train_dir):
    """Load labeled images and convert them into feature rows."""
    weed_dir = train_dir / "weed"
    not_weed_dir = train_dir / "not_weed"

    weed_images = get_image_paths(weed_dir)
    not_weed_images = get_image_paths(not_weed_dir)

    print(f"Weed images: {len(weed_images)} ({weed_dir})")
    print(f"Not-weed images: {len(not_weed_images)} ({not_weed_dir})")

    if not weed_images or not not_weed_images:
        raise RuntimeError(
            "training images are missing. "
            "Place weed images under images/train/weed/ and "
            "non-weed images under images/train/not_weed/."
        )

    samples = []
    labels = []
    skipped = []

    for label, image_paths in (
        (LABEL_WEED, weed_images),
        (LABEL_NOT_WEED, not_weed_images),
    ):
        for path in image_paths:
            image_bgr = cv2.imread(str(path))
            if image_bgr is None:
                skipped.append(path)
                continue
            samples.append(extract_features(image_bgr))
            labels.append(label)

    if not samples:
        raise RuntimeError("no readable training images found")

    return samples, labels, skipped


def train_model(train_dir, model_path, scaler_path):
    """Train and save the classifier."""
    print("=" * 60)
    print("Training model")
    print("=" * 60)

    samples, labels, skipped = collect_training_data(train_dir)

    if skipped:
        print(f"Skipped unreadable images: {len(skipped)}")
        for path in skipped:
            print(f"  - {path}")

    classifier = WeedClassifier()
    classifier.train(samples, labels)

    correct = 0
    for sample, label in zip(samples, labels):
        if classifier.predict(sample) == label:
            correct += 1
    train_accuracy = correct / len(labels)

    classifier.save(model_path, scaler_path)

    feature_dims = samples[0].shape[1]
    print(f"Feature matrix: {len(samples)} samples x {feature_dims} dims")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print("=" * 60)


def build_arg_parser():
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Train the weed classifier")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=TRAIN_DIR,
        help="path to the training root containing weed/ and not_weed/",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="output path for the OpenCV SVM model",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=SCALER_PATH,
        help="output path for the feature scaler",
    )
    return parser


def main():
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        train_model(args.train_dir, args.model_path, args.scaler_path)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
