import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import eval_detector


def test_detector_evaluation_uses_test_split_and_cli_paths(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pt"
    data_yaml = tmp_path / "detector.yaml"
    weights.write_bytes(b"weights")
    data_yaml.write_text("test: images/test\n", encoding="utf-8")
    calls = []

    class Box:
        map50 = map = mp = mr = 0.0

    class FakeModel:
        def __init__(self, path):
            assert path == str(weights)

        def val(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(box=Box())

        def predict(self, *args, **kwargs):
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_detector.py",
            "--weights",
            str(weights),
            "--data-yaml",
            str(data_yaml),
        ],
    )

    eval_detector.main()

    assert calls == [{"data": str(data_yaml), "split": "test", "imgsz": 640}]
