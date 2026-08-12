import argparse
import re
from collections import defaultdict
from pathlib import Path

from backend.app import config

_EAR_SUFFIX = re.compile(r"_v\d+$")


def ear_id(path: Path):
    return _EAR_SUFFIX.sub("", path.stem)


def validate_classifier(root, problems):
    root = Path(root)
    if not root.exists():
        problems.append(f"classifier root missing: {root}")
        return
    for variety_dir in sorted(root.iterdir()):
        if not variety_dir.is_dir():
            continue
        if variety_dir.name not in config.VARIETY_CLASSES:
            problems.append(f"unknown variety folder '{variety_dir.name}' (expected {config.VARIETY_CLASSES})")
            continue
        images = list(variety_dir.rglob("*"))
        n = len([p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if n == 0:
            problems.append(f"empty variety folder: {variety_dir.name}")


def validate_detector(img_root, lbl_root, problems):
    img_root = Path(img_root)
    lbl_root = Path(lbl_root)
    if not img_root.exists():
        problems.append(f"detector images root missing: {img_root}")
        return

    ears = defaultdict(lambda: {"images": 0, "labels": 0})
    for img in sorted(p for p in img_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        key = ear_id(img)
        ears[key]["images"] += 1
        label = lbl_root / f"{img.stem}.txt"
        if not label.exists():
            problems.append(f"missing label for {img.name}")
            continue
        ears[key]["labels"] += 1
        try:
            lines = label.read_text(encoding="utf-8").strip().splitlines()
        except OSError as exc:
            problems.append(f"unreadable label {label.name}: {exc}")
            continue
        has_ear = False
        for line_no, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{label.name}:{line_no} expected 5 fields, got {len(parts)}")
                continue
            cls, x, y, w, h = parts
            try:
                cls_idx = int(cls)
            except ValueError:
                problems.append(f"{label.name}:{line_no} class not an integer: {cls}")
                continue
            if not 0 <= cls_idx < len(config.DEFECT_CLASSES):
                problems.append(
                    f"{label.name}:{line_no} class {cls_idx} out of range 0..{len(config.DEFECT_CLASSES) - 1}"
                )
            for name, value in (("x", x), ("y", y), ("w", w), ("h", h)):
                try:
                    num = float(value)
                except ValueError:
                    problems.append(f"{label.name}:{line_no} coordinate {name} not numeric: {value}")
                    break
                if not 0.0 <= num <= 1.0:
                    problems.append(f"{label.name}:{line_no} coordinate {name} out of [0,1]: {num}")
            if cls_idx == 0:
                has_ear = True
        if not has_ear:
            problems.append(f"{label.name}: missing required class 0 (corn_ear) box")

    for key, info in ears.items():
        if info["images"] != info["labels"]:
            problems.append(f"ear '{key}': {info['images']} images vs {info['labels']} labels")


def main():
    parser = argparse.ArgumentParser(description="Validate the raw annotated dataset before training.")
    parser.add_argument("--classifier-raw", default="data/raw/classifier")
    parser.add_argument("--detector-images", default="data/raw/detector/images")
    parser.add_argument("--detector-labels", default="data/raw/detector/labels")
    args = parser.parse_args()

    problems = []
    validate_classifier(args.classifier_raw, problems)
    if Path(args.detector_images).exists():
        validate_detector(args.detector_images, args.detector_labels, problems)

    if problems:
        print(f"FOUND {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("Dataset OK.")


if __name__ == "__main__":
    main()