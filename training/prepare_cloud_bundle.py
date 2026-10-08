"""Build a validated Colab ZIP for classifier and detector experiments."""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

from backend.app import config
from training.make_dataset_split import COLAB_FIXED_DETECTOR_SPLITS, RATIOS
from training.validate_dataset import check_label_lines

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
EAR_VIEW = re.compile(r"^(ear\d{3})_v([1-4])$", re.IGNORECASE)
SEED = 42

_REQUIRED_FILES = (
    Path("training/__init__.py"),
    Path("training/make_dataset_split.py"),
    Path("training/materialize_cloud_classifier.py"),
    Path("training/relocate_yolo_dataset.py"),
    Path("training/validate_dataset.py"),
    Path("training/train_classifier.py"),
    Path("training/train_detector.py"),
    Path("training/yolo_comparison.py"),
    Path("eval/__init__.py"),
    Path("eval/eval_classifier.py"),
    Path("eval/eval_detector.py"),
    Path("backend/app/__init__.py"),
    Path("backend/app/config.py"),
)

_CLOUD_REQUIREMENTS = """\
torch>=2.2
torchvision>=0.17
ultralytics==8.4.155
numpy>=1.26
Pillow>=10
"""


def _image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing image directory: {directory}")
    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not path.is_file():
            raise ValueError(f"Image must be a regular file: {path}")
        stem = path.stem.lower()
        if stem in images:
            raise ValueError(f"Duplicate image stem in {directory}: {stem}")
        if not EAR_VIEW.fullmatch(stem):
            raise ValueError(f"Invalid image name {path.name}; expected earNNN_v1..v4")
        images[stem] = path
    if not images:
        raise ValueError(f"No images found in {directory}")
    return images


def _ear(stem: str) -> str:
    match = EAR_VIEW.fullmatch(stem)
    if match is None:
        raise ValueError(f"Invalid image stem: {stem}")
    return match.group(1).lower()


def _ear_views(images: dict[str, Path], description: str) -> set[str]:
    views: dict[str, set[int]] = {}
    for stem in images:
        match = EAR_VIEW.fullmatch(stem)
        assert match is not None  # _image_files checked this.
        views.setdefault(match.group(1).lower(), set()).add(int(match.group(2)))
    incomplete = sorted(ear for ear, numbers in views.items() if numbers != {1, 2, 3, 4})
    if incomplete:
        raise ValueError(f"{description} must have exactly four views per ear: {', '.join(incomplete)}")
    return set(views)


def _sources(root: Path) -> tuple[dict[str, dict[str, Path]], dict[str, Path], dict[str, Path]]:
    classifier: dict[str, dict[str, Path]] = {}
    ears_by_variety: dict[str, set[str]] = {}
    for variety in config.VARIETY_CLASSES:
        entries = _image_files(root / "data/raw/classifier" / variety)
        classifier[variety] = entries
        ears_by_variety[variety] = _ear_views(entries, variety)

    all_ears: set[str] = set()
    for ears in ears_by_variety.values():
        overlap = all_ears & ears
        if overlap:
            raise ValueError(f"Ear IDs appear in multiple varieties: {', '.join(sorted(overlap))}")
        all_ears.update(ears)

    detector = _image_files(root / "data/raw/detector/images")
    _ear_views(detector, "detector")
    classifier_stems = set().union(*(set(entries) for entries in classifier.values()))
    if set(detector) != classifier_stems:
        missing = sorted(classifier_stems - set(detector))
        extra = sorted(set(detector) - classifier_stems)
        raise ValueError(
            f"Detector and classifier image views are not exactly paired; missing={missing}, extra={extra}"
        )

    labels_root = root / "data/raw/detector/labels"
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Missing detector label directory: {labels_root}")
    label_paths: dict[str, Path] = {}
    for stem, detector_image in detector.items():
        source_image = next(entries[stem] for entries in classifier.values() if stem in entries)
        if source_image.read_bytes() != detector_image.read_bytes():
            raise ValueError(f"Classifier/detector image bytes differ: {stem}")
        label = labels_root / f"{detector_image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"Missing detector label: {label}")
        problems: list[str] = []
        check_label_lines(label.name, label.read_text(encoding="utf-8").splitlines(), problems)
        if problems:
            raise ValueError("; ".join(problems))
        label_paths[stem] = label
    return classifier, detector, label_paths


def _assign(ears_by_variety: dict[str, set[str]]) -> dict[str, str]:
    yellow = set(ears_by_variety["yellow_sweet_corn"])
    missing_fixed = sorted(set(COLAB_FIXED_DETECTOR_SPLITS) - yellow)
    if missing_fixed:
        raise ValueError(f"Fixed yellow ears are missing: {', '.join(missing_fixed)}")
    assignments = dict(COLAB_FIXED_DETECTOR_SPLITS)
    for variety in config.VARIETY_CLASSES:
        remaining = sorted(ears_by_variety[variety] - set(assignments))
        random.Random(SEED).shuffle(remaining)
        train_end = int(len(remaining) * RATIOS["train"])
        val_end = int(len(remaining) * (RATIOS["train"] + RATIOS["val"]))
        for split, group in zip(
            SPLITS, (remaining[:train_end], remaining[train_end:val_end], remaining[val_end:])
        ):
            assignments.update({ear: split for ear in group})
    for variety, ears in ears_by_variety.items():
        for split in SPLITS:
            if not any(assignments[ear] == split for ear in ears):
                raise ValueError(f"{variety} has no {split} ears; collect more data before bundling")
    return assignments


def _detector_yaml() -> str:
    lines = ["path: data/detector"]
    lines.extend(f"{split}: images/{split}" for split in SPLITS)
    lines.append(f"nc: {len(config.DEFECT_CLASSES)}")
    lines.append(f"names: {list(config.DEFECT_CLASSES)!r}")
    return "\n".join(lines) + "\n"


def _readme(manifest: dict) -> str:
    split_lines = "\n".join(
        f"- {split}: {manifest['splits'][split]['ear_count']} ears / "
        f"{manifest['splits'][split]['image_count']} images"
        for split in SPLITS
    )
    unrepresented = [
        name for name, count in manifest["detector_class_counts"].items() if count == 0
    ]
    absent_text = ", ".join(f"`{name}`" for name in unrepresented) or "none"
    absent_test = [
        name
        for name, count in manifest["splits"]["test"]["detector_class_counts"].items()
        if count == 0
    ]
    absent_test_text = ", ".join(f"`{name}`" for name in absent_test) or "none"
    return f"""# Maisagip combined Colab experiment

This archive contains the EfficientNetV2-S variety classifier and Ultralytics
detector experiment. Each image is stored once. Run the materialization step
below after extraction to create the classifier folders using local hardlinks
(or copies if hardlinks are unavailable). Both tasks use exactly the same ear-level split, so no
views of one ear cross train, validation, and test. Original assignments for
ear001–ear015 are preserved; remaining ears are assigned within each variety
with seed {SEED}. Exact IDs and class counts are in `manifest.json`.

{split_lines}

The detector taxonomy has {len(config.DEFECT_CLASSES)} classes. The current
snapshot has {manifest['detector_class_counts'].get('shriveled_kernels', 0)}
`shriveled_kernels` boxes. When this count is zero, the detector cannot learn
or be meaningfully evaluated for that defect. Rebuild the ZIP after annotating
class 7; previous weights do not gain the new class automatically.
Classes without any labeled boxes in this snapshot: {absent_text}. Results are
preliminary because the held-out sets contain only a few ears per variety.
Classes with no boxes in the held-out test split: {absent_test_text}. Test
metrics cannot establish performance on those classes. Per-split box counts
are recorded in `manifest.json`.

## In Google Colab

Select **Runtime → Change runtime type → GPU**. Upload this ZIP, then run:

```python
from google.colab import files
files.upload()
!unzip -q maisagip-cloud-joint-training-2026-10-08.zip -d /content/maisagip
%cd /content/maisagip
!pip install -r requirements-cloud.txt
```

If you renamed the ZIP, change its name in the unzip line. Run each command
below in a separate Colab code cell (prefix shell commands with `!`):

```bash
python -m training.relocate_yolo_dataset --data-yaml data/detector/data.yaml --dataset-root data/detector --in-place
python -m training.materialize_cloud_classifier --manifest manifest.json --data-root data
python -m training.train_classifier --data-root data/classifier --epochs 30 --batch-size 4 --save-to data/weights/efficientnetv2_s_corn.pt
python -m eval.eval_classifier --weights data/weights/efficientnetv2_s_corn.pt --data-root data/classifier/test
python -m training.yolo_comparison --models yolo11s --device 0 --epochs 100 --batch 16 --imgsz 640 --seed 42
```

The classifier evaluator prints accuracy and per-class/macro precision, recall,
and F1 on **test**. The YOLO comparison runner evaluates its best weights on
**test** and writes results to `data/comparisons/yolo/summary.csv`. For the
earlier four-model comparison, change `--models yolo11s` to
`--models yolov8n,yolov8s,yolov8m,yolo11s`.

Checkpoints and outputs live in Colab's temporary storage. Copy them to Drive
before the runtime ends if you want to keep them.
If classifier training still runs out of GPU memory, rerun its command with
`--batch-size 2` (or `1`). The classifier uses CUDA mixed precision automatically.
"""


def create_bundle(project_root: str | Path, output: str | Path) -> Path:
    """Create a portable ZIP from raw images without altering local splits."""
    root = Path(project_root).expanduser().resolve()
    destination = Path(output).expanduser()
    destination = (root / destination if not destination.is_absolute() else destination).resolve()
    for relative in _REQUIRED_FILES:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Missing required bundle file: {root / relative}")

    classifier, detector, labels = _sources(root)
    ears_by_variety = {
        variety: {_ear(stem) for stem in entries} for variety, entries in classifier.items()
    }
    assignments = _assign(ears_by_variety)
    class_counts = Counter()
    split_class_counts = {split: Counter() for split in SPLITS}
    for stem, label in labels.items():
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_name = config.DEFECT_CLASSES[int(line.split()[0])]
                class_counts[class_name] += 1
                split_class_counts[assignments[_ear(stem)]][class_name] += 1
    manifest = {
        "seed": SEED,
        "varieties": list(config.VARIETY_CLASSES),
        "ear_varieties": {
            ear: variety for variety, ears in ears_by_variety.items() for ear in ears
        },
        "detector_classes": list(config.DEFECT_CLASSES),
        "detector_class_counts": {name: class_counts[name] for name in config.DEFECT_CLASSES},
        "splits": {},
    }
    for split in SPLITS:
        ears = sorted(ear for ear, assigned in assignments.items() if assigned == split)
        manifest["splits"][split] = {
            "ears": ears,
            "ear_count": len(ears),
            "image_count": len(ears) * 4,
            "variety_ear_counts": {
                variety: sum(ear in ears for ear in variety_ears)
                for variety, variety_ears in ears_by_variety.items()
            },
            "detector_class_counts": {
                name: split_class_counts[split][name] for name in config.DEFECT_CLASSES
            },
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in _REQUIRED_FILES:
            archive.write(root / relative, relative.as_posix())
        for stem, source in detector.items():
            split = assignments[_ear(stem)]
            archive.write(source, f"data/detector/images/{split}/{source.name}")
            archive.write(labels[stem], f"data/detector/labels/{split}/{source.stem}.txt")
        archive.writestr("data/detector/data.yaml", _detector_yaml())
        archive.writestr("requirements-cloud.txt", _CLOUD_REQUIREMENTS)
        archive.writestr("CLOUD_README.md", _readme(manifest))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Maisagip joint Colab training ZIP.")
    parser.add_argument(
        "--output", type=Path, default=Path("maisagip-cloud-joint-training-2026-10-08.zip")
    )
    args = parser.parse_args()
    print(f"Created {create_bundle(Path(__file__).resolve().parents[1], args.output)}")


if __name__ == "__main__":
    main()
