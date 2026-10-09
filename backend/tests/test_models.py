import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image

from backend.app import config, models


def test_real_detector_passes_uploaded_rgb_images_as_pil(monkeypatch):
    calls = []

    class FakeYOLO:
        def __init__(self, weights_path):
            self.weights_path = weights_path

        def predict(self, source, **kwargs):
            calls.append((source, kwargs))
            return [SimpleNamespace(boxes=None)]

    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    detector = models.RealDetector("detector.pt")
    image = np.array(
        [
            [[255, 0, 10], [20, 40, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )

    assert detector.predict(image) == []
    assert len(calls) == 1

    source, kwargs = calls[0]
    assert isinstance(source, Image.Image)
    assert source.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(source), image)
    assert kwargs == {
        "imgsz": config.DETECTOR_IMAGE_SIZE,
        "conf": config.DEFAULT_CONFIDENCE_THRESHOLD,
        "iou": config.DETECTOR_NMS_IOU,
        "verbose": False,
    }


def test_demo_detector_filters_detections_by_request_threshold(monkeypatch):
    monkeypatch.setattr(models, "get_detector", lambda: ("demo", None))
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(
        models,
        "_demo_boxes",
        lambda image: [
            {"class": "corn_ear", "confidence": 0.51, "box": [0, 0, 10, 10]},
            {"class": "mold", "confidence": 0.49, "box": [1, 1, 5, 5]},
        ],
    )
    assert [d["class"] for d in models.detect_defects(image, 0.5)] == ["corn_ear"]
