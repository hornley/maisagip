from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import eval_classifier


def test_test_mapping_must_match_config(tmp_path):
    from PIL import Image

    root = tmp_path / "test"
    for name in ("white_corn", "yellow_sweet_corn"):
        directory = root / name
        directory.mkdir(parents=True)
        Image.new("RGB", (2, 2), color="white").save(directory / "sample.jpg")
    assert eval_classifier.validate_test_class_mapping(root) == {
        "white_corn": 0,
        "yellow_sweet_corn": 1,
    }

    (root / "white_corn").rename(root / "zebra")
    with pytest.raises(ValueError, match="class mapping"):
        eval_classifier.validate_test_class_mapping(root)


def test_metrics_report_unweighted_macro_values():
    # Class 0 is perfect; class 1 has one false positive and one miss.
    accuracy, precision, recall, f1 = eval_classifier.classification_metrics(
        [[2, 0], [1, 1]]
    )
    assert accuracy == pytest.approx(0.75)
    assert precision == pytest.approx((2 / 3 + 1.0) / 2)
    assert recall == pytest.approx((1.0 + 0.5) / 2)
    assert f1 == pytest.approx((0.8 + 2 / 3) / 2)
