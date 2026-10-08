from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import train_classifier


def _split(root, classes, files=True):
    for name in classes:
        class_dir = Path(root) / name
        class_dir.mkdir(parents=True, exist_ok=True)
        if files:
            (class_dir / "sample.jpg").write_bytes(b"not an image")


def test_validation_requires_configured_classes_in_each_split(tmp_path):
    split = tmp_path / "train"
    _split(split, ["white_corn", "yellow_sweet_corn"])

    # ImageFolder needs valid image data, so use a minimal real fixture here.
    from PIL import Image

    for path in split.glob("*/*.jpg"):
        Image.new("RGB", (2, 2), color="white").save(path)
    dataset = train_classifier.validate_classification_split(split)
    assert dataset.class_to_idx == {"white_corn": 0, "yellow_sweet_corn": 1}

    (split / "yellow_sweet_corn" / "sample.jpg").unlink()
    with pytest.raises(ValueError, match="no valid images"):
        train_classifier.validate_classification_split(split)


def test_validation_rejects_wrong_class_order_or_extra_class(tmp_path):
    split = tmp_path / "val"
    _split(split, ["white_corn", "yellow_sweet_corn", "other"])
    with pytest.raises(ValueError, match="exactly"):
        train_classifier.validate_classification_split(split)


def test_first_zero_accuracy_is_a_best_checkpoint():
    best_acc = None
    observed_acc = 0.0
    assert train_classifier.should_save_checkpoint(observed_acc, best_acc)


def test_batch_size_cli_defaults_to_four_and_rejects_non_positive_values():
    parser = train_classifier.build_parser()

    assert parser.parse_args([]).batch_size == 4
    assert parser.parse_args(["--batch-size", "7"]).batch_size == 7
    with pytest.raises(SystemExit):
        parser.parse_args(["--batch-size", "0"])


def test_make_loader_uses_requested_batch_size(tmp_path):
    from PIL import Image

    split = tmp_path / "train"
    _split(split, ["white_corn", "yellow_sweet_corn"])
    for path in split.glob("*/*.jpg"):
        Image.new("RGB", (2, 2), color="white").save(path)

    loader = train_classifier.make_loader(split, target_size=2, batch_size=1)

    assert loader.batch_size == 1
