"""Create a small ZIP bundle for cloud YOLO training.

The bundle contains the prepared detector dataset, the comparison runner, and
only the project files needed by that runner. It deliberately excludes raw
HEIC files, caches, virtual environments, model outputs, and local weights.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

_REQUIRED_FILES = (
    Path("training/__init__.py"),
    Path("training/make_dataset_split.py"),
    Path("training/relocate_yolo_dataset.py"),
    Path("training/yolo_comparison.py"),
    Path("backend/app/__init__.py"),
    Path("backend/app/config.py"),
)

_CLOUD_REQUIREMENTS = """\
ultralytics==8.4.155
numpy>=1.26
"""

_CLOUD_README = """\
# Maisagip cloud YOLO bundle

This archive contains the prepared detector dataset and the files needed for
the YOLO comparison runner. It contains normalized JPEG images and YOLO TXT
labels; the original HEIC files are intentionally excluded.

The detector split is fixed at the ear level:

```text
train: 22 ears / 88 images
val:    4 ears / 16 images
test:   5 ears / 20 images
```

The explicitly selected reject ears are `ear012, ear014` in train,
`ear013` in val, and `ear015` in test. Ears 16–31 were assigned with the
existing grouped 80/10/10 splitter using seed `42`. All views of one ear stay
in the same split, so the 124 images are not 124 independent samples.

## Colab

```bash
cd /content/maisagip
pip install -r requirements-cloud.txt
python -m training.relocate_yolo_dataset \\
  --data-yaml data/detector/data.yaml \\
  --dataset-root data/detector \\
  --in-place
python -m training.yolo_comparison \\
  --models yolov8n,yolov8s,yolov8m,yolo11s \\
  --device 0 \\
  --epochs 100 \\
  --batch 16 \\
  --imgsz 640 \\
  --seed 42
```

Ultralytics uses `val` during training and the held-out `test` split for the
final comparison after training. The results are preliminary because the test
set contains only five ears.

The same command works in Kaggle after changing into the extracted bundle
directory. On Kaggle, the directory is commonly `/kaggle/working/maisagip`.

The generated results are saved under `data/comparisons/yolo/`.
"""


def _add_file(archive: zipfile.ZipFile, source: Path, archive_name: Path) -> None:
    archive.write(source, archive_name.as_posix())


def _add_dataset(archive: zipfile.ZipFile, dataset_root: Path) -> None:
    for branch in ("images", "labels"):
        branch_root = dataset_root / branch
        for source in sorted(branch_root.rglob("*")):
            if not source.is_file():
                continue
            if branch == "images" and source.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue
            if branch == "labels" and source.suffix.lower() != ".txt":
                continue
            _add_file(archive, source, source.relative_to(dataset_root.parent.parent))


def create_bundle(project_root: str | Path, output: str | Path) -> Path:
    """Create a cloud-training ZIP and return its absolute path."""
    root = Path(project_root).expanduser().resolve()
    destination = Path(output).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    dataset_root = root / "data/detector"

    if not (dataset_root / "data.yaml").is_file():
        raise FileNotFoundError(f"Missing detector dataset: {dataset_root}")
    for relative in _REQUIRED_FILES:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Missing required bundle file: {root / relative}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in _REQUIRED_FILES:
            _add_file(archive, root / relative, relative)

        _add_dataset(archive, dataset_root)

        # Make the archive portable when extracted and run from its root.
        yaml_lines = (dataset_root / "data.yaml").read_text(
            encoding="utf-8"
        ).splitlines(keepends=True)
        path_lines = [
            index
            for index, line in enumerate(yaml_lines)
            if line.lstrip().startswith("path:")
        ]
        if len(path_lines) != 1:
            raise ValueError(
                "Expected exactly one path: entry in data/detector/data.yaml"
            )
        index = path_lines[0]
        newline = "\n" if yaml_lines[index].endswith("\n") else ""
        yaml_lines[index] = f"path: data/detector{newline}"
        yaml_text = "".join(yaml_lines)
        archive.writestr("data/detector/data.yaml", yaml_text)
        archive.writestr("requirements-cloud.txt", _CLOUD_REQUIREMENTS)
        archive.writestr("CLOUD_README.md", _CLOUD_README)

    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a cloud YOLO training ZIP.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maisagip-cloud-training.zip"),
        help="ZIP path; defaults to maisagip-cloud-training.zip",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    destination = create_bundle(Path(__file__).resolve().parents[1], args.output)
    print(f"Created {destination}")


if __name__ == "__main__":
    main()
