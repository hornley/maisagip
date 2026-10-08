from pathlib import Path

from training.validate_dataset import validate_classifier, validate_detector


def test_validate_classifier_flags_unknown_variety(tmp_path):
    root = tmp_path / "raw" / "classifier"
    problem = root / "purple_corn"
    problem.mkdir(parents=True)
    (problem / "e001_v1.jpg").write_bytes(b"x")
    issues = []
    validate_classifier(root, issues)
    assert issues


def test_validate_detector_accepts_valid_set(tmp_path):
    img_root = tmp_path / "images"
    lbl_root = tmp_path / "labels"
    img_root.mkdir(); lbl_root.mkdir()
    (img_root / "ear001_v1.jpg").write_bytes(b"x")
    (lbl_root / "ear001_v1.txt").write_text("0 0.5 0.5 0.8 0.8\n1 0.3 0.3 0.1 0.1\n")
    issues = []
    validate_detector(img_root, lbl_root, issues)
    assert issues == []


def test_validate_detector_accepts_ear_decay_class(tmp_path):
    img_root = tmp_path / "images"
    lbl_root = tmp_path / "labels"
    img_root.mkdir(); lbl_root.mkdir()
    (img_root / "ear001_v1.jpg").write_bytes(b"x")
    (lbl_root / "ear001_v1.txt").write_text("0 0.5 0.5 0.8 0.8\n6 0.3 0.3 0.1 0.1\n")
    issues = []

    validate_detector(img_root, lbl_root, issues)

    assert issues == []


def test_validate_detector_accepts_shriveled_kernels_class(tmp_path):
    img_root = tmp_path / "images"
    lbl_root = tmp_path / "labels"
    img_root.mkdir(); lbl_root.mkdir()
    (img_root / "ear001_v1.jpg").write_bytes(b"x")
    (lbl_root / "ear001_v1.txt").write_text("0 0.5 0.5 0.8 0.8\n7 0.3 0.3 0.1 0.1\n")
    issues = []

    validate_detector(img_root, lbl_root, issues)

    assert issues == []


def test_validate_detector_flags_missing_label(tmp_path):
    img_root = tmp_path / "images"
    lbl_root = tmp_path / "labels"
    img_root.mkdir(); lbl_root.mkdir()
    (img_root / "ear001_v1.jpg").write_bytes(b"x")
    issues = []
    validate_detector(img_root, lbl_root, issues)
    assert any("missing label" in issue for issue in issues)


def test_validate_detector_flags_out_of_range_class(tmp_path):
    img_root = tmp_path / "images"
    lbl_root = tmp_path / "labels"
    img_root.mkdir(); lbl_root.mkdir()
    (img_root / "ear001_v1.jpg").write_bytes(b"x")
    (lbl_root / "ear001_v1.txt").write_text("0 0.5 0.5 0.8 0.8\n9 0.3 0.3 0.1 0.1\n")
    issues = []
    validate_detector(img_root, lbl_root, issues)
    assert any("out of range" in issue for issue in issues)


def test_validate_detector_flags_missing_corn_ear_box(tmp_path):
    img_root = tmp_path / "images"
    lbl_root = tmp_path / "labels"
    img_root.mkdir(); lbl_root.mkdir()
    (img_root / "ear001_v1.jpg").write_bytes(b"x")
    (lbl_root / "ear001_v1.txt").write_text("2 0.5 0.5 0.1 0.1\n")
    issues = []
    validate_detector(img_root, lbl_root, issues)
    assert any("required class 0" in issue for issue in issues)


def test_validate_detector_rejects_zero_size_and_outside_boxes(tmp_path):
    img_root = tmp_path / "images"
    lbl_root = tmp_path / "labels"
    img_root.mkdir(); lbl_root.mkdir()
    (img_root / "ear001_v1.jpg").write_bytes(b"x")
    (lbl_root / "ear001_v1.txt").write_text("0 0.1 0.1 0 0.2\n1 0.95 0.5 0.2 0.2\n")
    issues = []

    validate_detector(img_root, lbl_root, issues)

    assert any("greater than zero" in issue for issue in issues)
    assert any("outside the image horizontally" in issue for issue in issues)


def test_validate_detector_accepts_rounding_at_image_edge(tmp_path):
    img_root = tmp_path / "images"
    lbl_root = tmp_path / "labels"
    img_root.mkdir(); lbl_root.mkdir()
    (img_root / "ear001_v1.jpg").write_bytes(b"x")
    (lbl_root / "ear001_v1.txt").write_text("0 0.4909 0.5314 0.3643 0.9373\n")
    issues = []

    validate_detector(img_root, lbl_root, issues)

    assert issues == []
