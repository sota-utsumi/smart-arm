"""
Weed detector using OpenCV.

Supported workflows:
- Detect weed regions in still images from images/source.
- Read camera settings from a config file and print the largest weed position
  in live camera frames.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
from weed_model import (
    HSV_LOWER,
    HSV_UPPER,
    LABEL_MAP,
    MIN_CONTOUR_AREA,
    MODEL_PATH,
    OUTPUT_DIR,
    SCALER_PATH,
    SOURCE_DIR,
    WeedClassifier,
    extract_features,
    get_image_paths,
)


def load_classifier():
    """Load the saved classifier."""
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("Run python learn.py first.")
        sys.exit(1)
    if not SCALER_PATH.exists():
        print(f"[ERROR] Scaler file not found: {SCALER_PATH}")
        print("Run python learn.py first.")
        sys.exit(1)

    print(f"Loading model: {MODEL_PATH}")
    print(f"Loading scaler: {SCALER_PATH}")
    try:
        return WeedClassifier().load(MODEL_PATH, SCALER_PATH)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def _contour_center(contour, bbox):
    """Return a contour center in image coordinates."""
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        return (center_x, center_y)

    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def detect_weed_regions(image_bgr):
    """
    Detect weed candidate regions with HSV segmentation.

    Returns a list sorted by area descending.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = image_bgr.shape[:2]
    total_area = img_h * img_w

    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        bbox = cv2.boundingRect(contour)
        regions.append(
            {
                "bbox": bbox,
                "area_px": int(area),
                "area_ratio": round(area / total_area * 100, 2),
                "center": _contour_center(contour, bbox),
                "contour": contour,
            }
        )

    regions.sort(key=lambda region: region["area_px"], reverse=True)
    return regions


def annotate_image(
    image_bgr,
    regions,
    classification_label="live",
    classification_score=None,
):
    """Draw detection results on an image."""
    annotated = image_bgr.copy()
    img_h, img_w = annotated.shape[:2]

    if classification_score is None:
        header_text = classification_label
    else:
        header_text = f"Classification: {classification_label} ({classification_score:.1%})"

    cv2.rectangle(annotated, (0, 0), (img_w, 40), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        header_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    total_weed_area = 0
    for index, region in enumerate(regions, start=1):
        x, y, w, h = region["bbox"]
        center_x, center_y = region["center"]
        total_weed_area += region["area_px"]

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        overlay = annotated.copy()
        cv2.drawContours(overlay, [region["contour"]], -1, (0, 200, 0), -1)
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

        cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)

        label = (
            f"#{index} center=({center_x},{center_y}) "
            f"bbox={w}x{h}px area={region['area_px']}px ({region['area_ratio']}%)"
        )
        label_y = max(y - 10, 50)
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (x, label_y - text_h - 4),
            (x + text_w + 4, label_y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    total_ratio = round(total_weed_area / (img_h * img_w) * 100, 2) if total_weed_area else 0
    footer = f"Weed regions: {len(regions)} | Total area: {total_weed_area}px ({total_ratio}%)"
    cv2.rectangle(annotated, (0, img_h - 35), (img_w, img_h), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        footer,
        (10, img_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return annotated


def process_source_images():
    """Run detection for all still images in images/source."""
    print("=" * 60)
    print("Processing source images")
    print("=" * 60)

    source_files = get_image_paths(SOURCE_DIR)
    if not source_files:
        print(f"[ERROR] No source images found in: {SOURCE_DIR}")
        sys.exit(1)

    print(f"Source images: {len(source_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv_path = OUTPUT_DIR / "detections.csv"

    classifier = load_classifier()
    csv_rows = []

    for index, img_path in enumerate(source_files, start=1):
        print(f"[{index}/{len(source_files)}] Processing {img_path.name} ...")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print("  -> failed to read image")
            continue

        feat = extract_features(image_bgr)
        pred, score = classifier.predict_with_confidence(feat)
        label = LABEL_MAP.get(pred, "unknown")

        print(f"  Classification: {label} (confidence: {score:.2%})")

        regions = detect_weed_regions(image_bgr)
        print(f"  Weed regions: {len(regions)}")

        if regions:
            largest = regions[0]
            print(
                "  Largest weed center: "
                f"({largest['center'][0]}, {largest['center'][1]}) px"
            )
            csv_rows.append(
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "x": largest["center"][0],
                    "y": largest["center"][1],
                    "size": largest["area_px"],
                }
            )
        else:
            csv_rows.append(
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "x": "",
                    "y": "",
                    "size": 0,
                }
            )

        for region_index, region in enumerate(regions, start=1):
            x, y, w, h = region["bbox"]
            center_x, center_y = region["center"]
            print(
                f"    #{region_index}: center=({center_x}, {center_y}) px, "
                f"bbox=({x}, {y}, {w}, {h}), "
                f"area={region['area_px']}px ({region['area_ratio']}%)"
            )

        print()

    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["timestamp", "x", "y", "size"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("=" * 60)
    print(f"Done. Detection CSV was saved to: {output_csv_path}")
    print("=" * 60)


def _parse_camera_source(raw_value):
    """Convert a numeric camera source to int, keep device paths as strings."""
    value = raw_value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def load_camera_config(config_path):
    """Load camera settings from a key=value config file."""
    path = Path(config_path)
    if not path.exists():
        print(f"[ERROR] Config file not found: {path}")
        sys.exit(1)

    config = {}
    try:
        with path.open("r", encoding="utf-8") as config_file:
            for line_number, raw_line in enumerate(config_file, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    raise ValueError(
                        f"Invalid config line {line_number}: {raw_line.rstrip()}"
                    )
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    raise ValueError(f"Empty key at line {line_number}")
                config[key] = value
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    required_keys = ("camera", "camera_width", "camera_height")
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"[ERROR] Missing config keys: {', '.join(missing_keys)}")
        sys.exit(1)

    try:
        camera_width = int(config["camera_width"])
        camera_height = int(config["camera_height"])
    except ValueError:
        print("[ERROR] camera_width and camera_height must be integers.")
        sys.exit(1)

    if camera_width <= 0 or camera_height <= 0:
        print("[ERROR] camera_width and camera_height must be positive.")
        sys.exit(1)

    return {
        "config_path": path,
        "camera_raw": config["camera"],
        "camera": _parse_camera_source(config["camera"]),
        "camera_width": camera_width,
        "camera_height": camera_height,
    }


def open_camera(camera_config):
    """Open the configured camera and apply the requested frame size."""
    capture = cv2.VideoCapture(camera_config["camera"])
    if not capture.isOpened():
        print(f"[ERROR] Failed to open camera: {camera_config['camera_raw']}")
        sys.exit(1)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["camera_width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["camera_height"])
    return capture


def _print_largest_weed_console(region):
    """Continuously update the console with the largest weed position."""
    if region is None:
        print("\rLargest weed center: not detected", end="", flush=True)
        return

    center_x, center_y = region["center"]
    x, y, w, h = region["bbox"]
    print(
        "\r"
        f"Largest weed center: ({center_x}, {center_y}) px | "
        f"bbox=({x}, {y}, {w}, {h}) | area={region['area_px']}px      ",
        end="",
        flush=True,
    )


def process_camera_stream(config_path):
    """Read frames from the configured camera and print the largest weed center."""
    camera_config = load_camera_config(config_path)

    print("=" * 60)
    print("Starting live camera detection")
    print("=" * 60)
    print(f"Config: {camera_config['config_path']}")
    print(f"Camera: {camera_config['camera_raw']}")
    print(
        "Requested size: "
        f"{camera_config['camera_width']}x{camera_config['camera_height']}"
    )

    capture = open_camera(camera_config)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width > 0 and actual_height > 0:
        print(f"Opened size: {actual_width}x{actual_height}")

    print("Press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("\n[ERROR] Failed to read a frame from the camera.")
                break

            regions = detect_weed_regions(frame)
            largest = regions[0] if regions else None
            _print_largest_weed_console(largest)
    except KeyboardInterrupt:
        print("\nStopped live camera detection.")
    finally:
        capture.release()
        print()


def build_arg_parser():
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Weed detector")
    parser.add_argument(
        "--detect",
        action="store_true",
        help="process still images from images/source/",
    )
    parser.add_argument(
        "--camera-detect",
        action="store_true",
        help="read the camera from --config and print the largest weed center",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="path to a config.txt file with camera settings",
    )
    return parser


def main():
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if not any([args.detect, args.camera_detect]):
        print("Examples:")
        print("  python weed_detector.py --detect")
        print("  python weed_detector.py --camera-detect --config config.txt")
        print()
        parser.print_help()
        sys.exit(0)

    if args.detect:
        process_source_images()

    if args.camera_detect:
        if args.config is None:
            parser.error("--camera-detect requires --config PATH")
        process_camera_stream(args.config)


if __name__ == "__main__":
    main()
