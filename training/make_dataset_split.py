import argparse
import random
import re
import shutil
from pathlib import Path

from backend.app import config

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

_EAR_SUFFIX = re.compile(r"_v\d+$")


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


def split_classifier(raw_root, out_root, seed):
    counts = {}
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    for variety_dir in sorted(raw_root.iterdir()):
        if not variety_dir.is_dir():
            continue
        files = sorted(p for p in variety_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
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
    imgs = sorted(p for p in Path(raw_images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        return {}
    splits = _split_groups(group_by_ear(imgs), seed)
    out_root = Path(out_root)
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


def main():
    parser = argparse.ArgumentParser(
        description="Create leak-free per-ear train/val/test splits for Maisagip."
    )
    parser.add_argument("--classifier-raw", default="data/raw/classifier")
    parser.add_argument("--detector-images", default="data/raw/detector/images")
    parser.add_argument("--detector-labels", default="data/raw/detector/labels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cls_counts = split_classifier(args.classifier_raw, config.DATA_DIR, args.seed)
    det_counts = {}
    if Path(args.detector_images).exists():
        det_counts = split_detector(
            args.detector_images, args.detector_labels, config.DATA_DIR / "detector", args.seed
        )

    print("Classifier split (images per variety):", cls_counts)
    print("Detector split (images):", det_counts)


if __name__ == "__main__":
    main()