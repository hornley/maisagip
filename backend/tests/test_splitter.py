from pathlib import Path

import pytest

from training.make_dataset_split import (
    COLAB_FIXED_DETECTOR_SPLITS,
    ear_id,
    group_by_ear,
    split_classifier,
    split_detector,
    split_detector_fixed,
)


def test_group_by_ear_splits_view_groups():
    files = [
        Path("ear001_v1.jpg"),
        Path("ear001_v2.jpg"),
        Path("ear002_v1.jpg"),
        Path("ear002_v2.jpg"),
    ]
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


def _make_fixed_detector_fixture(tmp_path, *, missing_ears=()):
    image_root = tmp_path / "raw" / "detector" / "images"
    label_root = tmp_path / "raw" / "detector" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    missing_ears = set(missing_ears)
    for ear in range(1, 32):
        if ear in missing_ears:
            continue
        for view in range(1, 3):
            stem = f"ear{ear:03d}_v{view}"
            (image_root / f"{stem}.jpg").write_bytes(b"x")
            (label_root / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.8 0.8\n", encoding="utf-8"
            )
    return image_root, label_root


def _split_ear_ids(root):
    return {
        split: {
            ear_id(path)
            for path in (root / "images" / split).iterdir()
            if path.is_file()
        }
        for split in ("train", "val", "test")
    }


def test_fixed_detector_split_uses_approved_ear_mapping_and_pairs_labels(tmp_path):
    image_root, label_root = _make_fixed_detector_fixture(tmp_path)
    out = tmp_path / "out" / "detector"

    counts = split_detector_fixed(image_root, label_root, out, seed=42)

    assert counts == {"train": 44, "val": 8, "test": 10}
    split_ears = _split_ear_ids(out)
    assert split_ears == {
        "train": {
            "ear003", "ear004", "ear005", "ear006", "ear007", "ear008",
            "ear009", "ear010", "ear012", "ear014", "ear017", "ear018",
            "ear021", "ear022", "ear023", "ear024", "ear025", "ear026",
            "ear028", "ear029", "ear030", "ear031",
        },
        "val": {"ear001", "ear013", "ear020", "ear027"},
        "test": {"ear002", "ear011", "ear015", "ear016", "ear019"},
    }
    assert set(COLAB_FIXED_DETECTOR_SPLITS) <= set.union(*split_ears.values())
    assert all(
        len([split for split, ears in split_ears.items() if ear in ears]) == 1
        for ear in {f"ear{number:03d}" for number in range(1, 32)}
    )
    for split in ("train", "val", "test"):
        images = sorted((out / "images" / split).iterdir())
        labels = sorted((out / "labels" / split).iterdir())
        assert len(images) == len(labels) == counts[split]
        assert [image.stem for image in images] == [label.stem for label in labels]


def test_fixed_detector_split_rejects_missing_required_ear(tmp_path):
    image_root, label_root = _make_fixed_detector_fixture(tmp_path, missing_ears=(15,))

    with pytest.raises(ValueError, match="ear015"):
        split_detector_fixed(image_root, label_root, tmp_path / "out", seed=42)


def test_fixed_detector_split_clears_stale_files_on_rerun(tmp_path):
    image_root, label_root = _make_fixed_detector_fixture(tmp_path)
    out = tmp_path / "out" / "detector"
    split_detector_fixed(image_root, label_root, out, seed=42)

    stale_image = out / "images" / "test" / "stale.jpg"
    stale_label = out / "labels" / "test" / "stale.txt"
    stale_image.write_bytes(b"stale")
    stale_label.write_text("stale", encoding="utf-8")

    split_detector_fixed(image_root, label_root, out, seed=42)

    assert not stale_image.exists()
    assert not stale_label.exists()
    assert _split_ear_ids(out)["test"] == {
        "ear002", "ear011", "ear015", "ear016", "ear019"
    }


def test_rerunning_splits_clears_stale_files_and_keeps_ears_isolated(tmp_path):
    classifier_raw = tmp_path / "raw" / "classifier"
    variety_dirs = [classifier_raw / "white_corn", classifier_raw / "yellow_sweet_corn"]
    for variety_dir in variety_dirs:
        variety_dir.mkdir(parents=True)
        for ear in range(10):
            for view in range(1, 3):
                (variety_dir / f"ear{ear:03d}_v{view}.jpg").write_bytes(b"x")

    detector_images = tmp_path / "raw" / "detector" / "images"
    detector_labels = tmp_path / "raw" / "detector" / "labels"
    detector_images.mkdir(parents=True)
    detector_labels.mkdir(parents=True)
    for ear in range(10):
        for view in range(1, 3):
            (detector_images / f"ear{ear:03d}_v{view}.jpg").write_bytes(b"x")
            (detector_labels / f"ear{ear:03d}_v{view}.txt").write_text("label")

    out = tmp_path / "out"
    split_classifier(classifier_raw, out, seed=42)
    split_detector(detector_images, detector_labels, out / "detector", seed=42)

    for variety_dir in variety_dirs:
        for ear in range(5, 10):
            for view in range(1, 3):
                (variety_dir / f"ear{ear:03d}_v{view}.jpg").unlink()
    for ear in range(5, 10):
        for view in range(1, 3):
            (detector_images / f"ear{ear:03d}_v{view}.jpg").unlink()
            (detector_labels / f"ear{ear:03d}_v{view}.txt").unlink()

    split_classifier(classifier_raw, out, seed=99)
    split_detector(detector_images, detector_labels, out / "detector", seed=99)

    for variety_dir in variety_dirs:
        split_ears = {
            split: {
                ear_id(path)
                for path in (out / "classifier" / split / variety_dir.name).iterdir()
            }
            for split in ("train", "val", "test")
        }
        assert all(
            f"ear{ear:03d}" not in split_ears[split]
            for ear in range(5, 10)
            for split in split_ears
        )
        for ear in range(5):
            assert sum(f"ear{ear:03d}" in split_ears[split] for split in split_ears) == 1

    detector_split_ears = {
        split: {
            ear_id(path)
            for path in (out / "detector" / "images" / split).iterdir()
        }
        for split in ("train", "val", "test")
    }
    detector_label_splits = {
        split: {path.stem for path in (out / "detector" / "labels" / split).iterdir()}
        for split in ("train", "val", "test")
    }
    assert all(
        f"ear{ear:03d}" not in detector_split_ears[split]
        for ear in range(5, 10)
        for split in detector_split_ears
    )
    assert all(
        f"ear{ear:03d}_v{view}" not in detector_label_splits[split]
        for ear in range(5, 10)
        for view in range(1, 3)
        for split in detector_label_splits
    )
    for ear in range(5):
        assert (
            sum(
                f"ear{ear:03d}" in detector_split_ears[split]
                for split in detector_split_ears
            )
            == 1
        )


@pytest.mark.parametrize("target_type", ["symlink", "file"])
def test_classifier_cleanup_rejects_unsafe_targets_before_deleting(tmp_path, target_type):
    raw = tmp_path / "raw" / "classifier" / "white_corn"
    raw.mkdir(parents=True)
    (raw / "ear001_v1.jpg").write_bytes(b"x")

    out = tmp_path / "out"
    safe_target = out / "classifier" / "train"
    safe_target.mkdir(parents=True)
    stale_file = safe_target / "stale.jpg"
    stale_file.write_bytes(b"stale")
    unsafe_target = out / "classifier" / "val"
    unsafe_target.parent.mkdir(parents=True, exist_ok=True)
    if target_type == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        unsafe_target.symlink_to(outside, target_is_directory=True)
    else:
        unsafe_target.write_bytes(b"not a directory")

    with pytest.raises(ValueError):
        split_classifier(raw.parent, out, seed=42)

    assert stale_file.exists()
    if target_type == "symlink":
        assert unsafe_target.is_symlink()
    else:
        assert unsafe_target.is_file()


@pytest.mark.parametrize("target_type", ["symlink", "file"])
def test_detector_cleanup_rejects_unsafe_targets_before_deleting(tmp_path, target_type):
    images = tmp_path / "raw" / "images"
    labels = tmp_path / "raw" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "ear001_v1.jpg").write_bytes(b"x")

    out = tmp_path / "out"
    safe_target = out / "images" / "train"
    safe_target.mkdir(parents=True)
    stale_file = safe_target / "stale.jpg"
    stale_file.write_bytes(b"stale")
    unsafe_target = out / "labels" / "val"
    unsafe_target.parent.mkdir(parents=True, exist_ok=True)
    if target_type == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        unsafe_target.symlink_to(outside, target_is_directory=True)
    else:
        unsafe_target.write_bytes(b"not a directory")

    with pytest.raises(ValueError):
        split_detector(images, labels, out, seed=42)

    assert stale_file.exists()
    if target_type == "symlink":
        assert unsafe_target.is_symlink()
    else:
        assert unsafe_target.is_file()


@pytest.mark.parametrize("layout", ["symlinked_root", "symlinked_ancestor"])
def test_cleanup_rejects_symlinked_output_paths_before_deleting(tmp_path, layout):
    raw = tmp_path / "raw" / "classifier" / "white_corn"
    raw.mkdir(parents=True)
    (raw / "ear001_v1.jpg").write_bytes(b"x")

    real_output = tmp_path / "real-output"
    if layout == "symlinked_root":
        out = tmp_path / "out-link"
        out.symlink_to(real_output, target_is_directory=True)
        cleanup_root = real_output
    else:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        out_parent = tmp_path / "parent-link"
        out_parent.symlink_to(real_parent, target_is_directory=True)
        out = out_parent / "out"
        cleanup_root = real_parent / "out"

    stale_file = cleanup_root / "classifier" / "train" / "stale.jpg"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"stale")

    with pytest.raises(ValueError):
        split_classifier(raw.parent, out, seed=42)

    assert stale_file.exists()


def test_split_rejects_an_output_root_that_is_a_generated_target(tmp_path):
    raw = tmp_path / "raw" / "classifier"
    raw.mkdir(parents=True)

    with pytest.raises(ValueError):
        split_classifier(raw, tmp_path / "out" / "classifier" / "train", seed=42)
