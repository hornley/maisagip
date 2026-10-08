import argparse
import os
import random
import re
import shutil
import stat
from pathlib import Path

from backend.app import config

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

_EAR_SUFFIX = re.compile(r"_v\d+$")
_SPLITS = ("train", "val", "test")
COLAB_FIXED_DETECTOR_SPLITS = {
    "ear001": "val",
    "ear002": "test",
    "ear003": "train",
    "ear004": "train",
    "ear005": "train",
    "ear006": "train",
    "ear007": "train",
    "ear008": "train",
    "ear009": "train",
    "ear010": "train",
    "ear011": "test",
    "ear012": "train",
    "ear013": "val",
    "ear014": "train",
    "ear015": "test",
}
_AMBIGUOUS_OUT_ROOT_SUFFIXES = {
    ("classifier", split) for split in _SPLITS
} | {
    ("detector", branch, split)
    for branch in ("images", "labels")
    for split in _SPLITS
} | {
    (branch, split)
    for branch in ("images", "labels")
    for split in _SPLITS
}


def ear_id(path: Path):
    return _EAR_SUFFIX.sub("", path.stem)


def group_by_ear(files):
    groups = {}
    for f in files:
        groups.setdefault(ear_id(f), []).append(f)
    return list(groups.values())


def _split_groups(groups, seed):
    random.seed(seed)
    random.shuffle(groups)
    n = len(groups)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    t1 = int(n * RATIOS["train"])
    t2 = int(n * (RATIOS["train"] + RATIOS["val"]))
    return {"train": groups[:t1], "val": groups[t1:t2], "test": groups[t2:]}


def _preflight_out_root(out_root):
    """Return an absolute output path after checking every existing component."""
    out_root = Path(out_root)
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root

    current = Path(out_root.anchor)
    parts = out_root.parts[1:]
    missing_component = False
    for index, part in enumerate(parts):
        if part == ".":
            continue
        if part == "..":
            current = current.parent
            continue

        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            missing_component = True
            break
        except NotADirectoryError as exc:
            raise ValueError(
                f"Split output path contains a non-directory ancestor: {current}"
            ) from exc

        if stat.S_ISLNK(mode):
            raise ValueError(f"Split output path contains a symlink: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise ValueError(
                f"Split output path contains a non-directory ancestor: {current}"
            )

    if not missing_component:
        out_root = Path(os.path.abspath(out_root))
    if out_root.exists() and not out_root.is_dir():
        raise ValueError(f"Split output root is not a directory: {out_root}")
    return out_root


def _clear_split_dirs(out_root, *relative_directories):
    """Clear generated split directories after validating the full cleanup set."""
    out_root = _preflight_out_root(out_root)
    if any(
        tuple(out_root.parts[-len(suffix) :]) == suffix
        for suffix in _AMBIGUOUS_OUT_ROOT_SUFFIXES
    ):
        raise ValueError(f"Ambiguous split output root: {out_root}")
    if out_root.exists() and not out_root.is_dir():
        raise ValueError(f"Split output root is not a directory: {out_root}")

    directories = []
    for relative_directory in relative_directories:
        relative_directory = Path(relative_directory)
        if (
            not relative_directory.parts
            or relative_directory.is_absolute()
            or ".." in relative_directory.parts
        ):
            raise ValueError(f"Invalid split cleanup target: {relative_directory}")

        directory = out_root / relative_directory
        if directory == out_root:
            raise ValueError(f"Split cleanup target is the output root: {directory}")
        try:
            directory.relative_to(out_root)
        except ValueError as exc:
            raise ValueError(
                f"Split cleanup target is outside output root: {directory}"
            ) from exc

        for parent in directory.parents:
            if parent == out_root:
                break
            if parent.is_symlink():
                raise ValueError(f"Split cleanup path contains a symlink: {parent}")
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Split cleanup path is not a directory: {parent}")

        if directory.is_symlink():
            raise ValueError(f"Refusing to remove symlink split target: {directory}")
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"Refusing to remove non-directory split target: {directory}")

        resolved_directory = directory.resolve()
        try:
            resolved_directory.relative_to(out_root)
        except ValueError as exc:
            raise ValueError(f"Split cleanup target escapes output root: {directory}") from exc
        directories.append(directory)

    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)


def split_classifier(raw_root, out_root, seed):
    counts = {}
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    _clear_split_dirs(
        out_root, *(Path("classifier") / split for split in _SPLITS)
    )
    for variety_dir in sorted(raw_root.iterdir()):
        if not variety_dir.is_dir():
            continue
        files = sorted(
            p
            for p in variety_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not files:
            continue
        splits = _split_groups(group_by_ear(files), seed)
        per_split = {}
        for split, groups in splits.items():
            dest = out_root / "classifier" / split / variety_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            total = 0
            for group in groups:
                for src in group:
                    shutil.copy2(src, dest / src.name)
                    total += 1
            per_split[split] = total
        counts[variety_dir.name] = per_split
    return counts


def split_detector(raw_images, raw_labels, out_root, seed):
    out_root = Path(out_root)
    _clear_split_dirs(
        out_root,
        *(Path("images") / split for split in _SPLITS),
        *(Path("labels") / split for split in _SPLITS),
    )
    imgs = sorted(
        p
        for p in Path(raw_images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not imgs:
        return {}
    splits = _split_groups(group_by_ear(imgs), seed)
    counts = {}
    for split, groups in splits.items():
        img_out = out_root / "images" / split
        lbl_out = out_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        total = 0
        for group in groups:
            for img in group:
                shutil.copy2(img, img_out / img.name)
                label = Path(raw_labels) / f"{img.stem}.txt"
                if label.exists():
                    shutil.copy2(label, lbl_out / label.name)
                total += 1
        counts[split] = total
    return counts


def split_detector_fixed(raw_images, raw_labels, out_root, seed=42):
    """Create the approved fixed-plus-seeded detector split for Colab.

    Ears 1 through 15 use ``COLAB_FIXED_DETECTOR_SPLITS``. Every other ear is
    assigned with the existing seeded 80/10/10 group split, keeping all of its
    views together. Unlike the generic splitter, this profile requires every
    image to have a same-stem YOLO label because it is intended for a complete
    cloud-training bundle.
    """
    out_root = Path(out_root)
    _clear_split_dirs(
        out_root,
        *(Path("images") / split for split in _SPLITS),
        *(Path("labels") / split for split in _SPLITS),
    )
    images = sorted(
        path
        for path in Path(raw_images).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No detector images found in {raw_images}")

    groups = {}
    for image in images:
        groups.setdefault(ear_id(image), []).append(image)

    missing_fixed = sorted(set(COLAB_FIXED_DETECTOR_SPLITS) - set(groups))
    if missing_fixed:
        missing_text = ", ".join(missing_fixed)
        raise ValueError(
            "Fixed detector split is missing required source ears: "
            f"{missing_text}"
        )

    labels_root = Path(raw_labels)
    missing_labels = sorted(
        image.stem
        for image in images
        if not (labels_root / f"{image.stem}.txt").is_file()
    )
    if missing_labels:
        missing_text = ", ".join(missing_labels)
        raise ValueError(
            "Fixed detector split is missing labels for image stems: "
            f"{missing_text}"
        )

    assignments = dict(COLAB_FIXED_DETECTOR_SPLITS)
    remaining_groups = [
        groups[ear]
        for ear in sorted(groups)
        if ear not in COLAB_FIXED_DETECTOR_SPLITS
    ]
    for split, split_groups in _split_groups(remaining_groups, seed).items():
        for group in split_groups:
            assignments[ear_id(group[0])] = split

    missing_assignments = sorted(set(groups) - set(assignments))
    if missing_assignments:
        missing_text = ", ".join(missing_assignments)
        raise ValueError(
            "Fixed detector split could not assign source ears: " f"{missing_text}"
        )

    counts = {split: 0 for split in _SPLITS}
    for split in _SPLITS:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for ear in sorted(groups):
        split = assignments[ear]
        image_destination = out_root / "images" / split
        label_destination = out_root / "labels" / split
        for image in groups[ear]:
            shutil.copy2(image, image_destination / image.name)
            label = labels_root / f"{image.stem}.txt"
            shutil.copy2(label, label_destination / label.name)
            counts[split] += 1
    return counts


def detector_split_summary(root):
    """Return sorted ear IDs present in each generated detector split."""
    root = Path(root)
    return {
        split: sorted(
            {
                ear_id(path)
                for path in (root / "images" / split).iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            }
        )
        for split in _SPLITS
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create leak-free per-ear train/val/test splits for Maisagip."
    )
    parser.add_argument("--classifier-raw", default="data/raw/classifier")
    parser.add_argument("--detector-images", default="data/raw/detector/images")
    parser.add_argument("--detector-labels", default="data/raw/detector/labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--detector-split-profile",
        choices=("random", "colab-fixed"),
        default="random",
        help="Detector split policy; colab-fixed preserves the approved ear IDs.",
    )
    args = parser.parse_args()

    cls_counts = split_classifier(args.classifier_raw, config.DATA_DIR, args.seed)
    det_counts = {}
    if Path(args.detector_images).exists():
        if args.detector_split_profile == "colab-fixed":
            det_counts = split_detector_fixed(
                args.detector_images,
                args.detector_labels,
                config.DATA_DIR / "detector",
                args.seed,
            )
        else:
            det_counts = split_detector(
                args.detector_images,
                args.detector_labels,
                config.DATA_DIR / "detector",
                args.seed,
            )

    print("Classifier split (images per variety):", cls_counts)
    print("Detector split (images):", det_counts)
    if det_counts and args.detector_split_profile == "colab-fixed":
        print("Detector split (ears):", detector_split_summary(config.DATA_DIR / "detector"))


if __name__ == "__main__":
    main()
