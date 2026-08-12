from pathlib import Path

from training.make_dataset_split import ear_id, group_by_ear, split_classifier, split_detector


def test_group_by_ear_splits_view_groups():
    files = [Path("ear001_v1.jpg"), Path("ear001_v2.jpg"), Path("ear002_v1.jpg"), Path("ear002_v2.jpg")]
    groups = group_by_ear(files)
    assert len(groups) == 2
    assert all(len(g) == 2 for g in groups)


def test_classifier_splits_keep_ears_together(tmp_path):
    raw = tmp_path / "raw" / "classifier" / "white_corn"
    raw.mkdir(parents=True)
    for ear in range(10):
        for view in range(1, 5):
            (raw / f"ear{ear:03d}_v{view}.jpg").write_bytes(b"x")

    out = tmp_path / "out"
    counts = split_classifier(tmp_path / "raw" / "classifier", out, seed=42)
    assert counts["white_corn"] == {"train": 32, "val": 4, "test": 4}

    split_ears = {}
    for split in ("train", "val", "test"):
        split_ears[split] = {
            ear_id(f) for f in (out / "classifier" / split / "white_corn").iterdir()
        }
    all_ears = set()
    for split in ("train", "val", "test"):
        all_ears |= split_ears[split]
    assert len(all_ears) == 10
    for ear in range(10):
        key = f"ear{ear:03d}"
        present = [split for split in ("train", "val", "test") if key in split_ears[split]]
        assert len(present) == 1, f"{key} leaked across splits: {present}"


def test_detector_splits_pair_images_and_labels(tmp_path):
    img_root = tmp_path / "raw" / "images"
    lbl_root = tmp_path / "raw" / "labels"
    img_root.mkdir(parents=True)
    lbl_root.mkdir(parents=True)
    for ear in range(6):
        for view in range(1, 5):
            (img_root / f"ear{ear:03d}_v{view}.jpg").write_bytes(b"x")
            (lbl_root / f"ear{ear:03d}_v{view}.txt").write_text("0 0.5 0.5 0.8 0.8\n")

    out = tmp_path / "out"
    counts = split_detector(img_root, lbl_root, out, seed=7)
    assert sum(counts.values()) == 24
    for split, n in counts.items():
        images = list((out / "images" / split).iterdir())
        labels = list((out / "labels" / split).iterdir())
        assert len(images) == n
        assert len(labels) == n
        assert all(i.stem == l.stem for i, l in zip(sorted(images), sorted(labels)))