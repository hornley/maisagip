from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import config, image_io, pipeline
from backend.app.main import app

client = TestClient(app)


def make_corn_image(defective=False):
    w, h = 600, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)
    x = np.linspace(0, 1, h)[:, None]
    img[..., 0] = (230 + 20 * x).clip(0, 255).astype(np.uint8)
    img[..., 1] = (180 + 20 * x).clip(0, 255).astype(np.uint8)
    img[..., 2] = (60 + 20 * x).clip(0, 255).astype(np.uint8)
    if defective:
        img[260:360, 230:330] = 60
    return img


def png_bytes(array):
    buf = BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def upload(files):
    payload = [("files", (f"v{i + 1}.png", png_bytes(img), "image/png")) for i, img in enumerate(files)]
    return client.post("/inspect", files=payload)


def test_decode_image_delegates_to_image_io_and_provider_gets_rgb_array(monkeypatch):
    source = b"encoded image bytes"
    decoded = np.zeros((2, 3, 3), dtype=np.uint8)
    decode_calls = []
    provider_images = []

    def fake_decode_image_bytes(data):
        decode_calls.append(data)
        return decoded

    def fake_classify_variety(image):
        provider_images.append(image)
        return [{"class": "yellow_sweet_corn", "confidence": 0.95}]

    monkeypatch.setattr(image_io, "decode_image_bytes", fake_decode_image_bytes)
    monkeypatch.setattr(pipeline.models, "classify_variety", fake_classify_variety)
    monkeypatch.setattr(pipeline.models, "detect_defects", lambda image: [])
    monkeypatch.setattr(pipeline.models, "provider_modes", lambda: {})

    report = pipeline.inspect_ear([source])

    assert decode_calls == [source]
    assert len(provider_images) == 1
    assert provider_images[0] is decoded
    assert isinstance(provider_images[0], np.ndarray)
    assert provider_images[0].shape[-1] == 3
    assert report["view_count"] == 1


def test_health_reports_providers():
    res = client.get("/health")
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "ok"


def test_single_image_inspection_still_works():
    res = upload([make_corn_image()])
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["view_count"] == 1
    assert report["variety"]["class"] in config.VARIETY_CLASSES
    assert "defect_coverage" in report
    assert report["grade"]["grade_label"]
    assert report["image_url"].startswith("/report/")


def test_multi_view_inspection_reports_views_and_merge():
    res = upload([make_corn_image(), make_corn_image(), make_corn_image(), make_corn_image()])
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["view_count"] == 4
    assert len(report["views"]) == 4
    assert all(v["image_url"].startswith("/report/") for v in report["views"])
    assert report["defect_coverage"] == 0.0
    assert report["grade"]["grade"] == "ExtraClass"


def test_duplicate_defect_across_two_views_merges_to_one(monkeypatch):
    monkeypatch.setattr(
        pipeline.models,
        "classify_variety",
        lambda image: [{"class": "yellow_sweet_corn", "confidence": 0.95}],
    )

    def fake_detect_defects(image):
        h, w = image.shape[:2]
        detections = [
            {
                "class": "corn_ear",
                "confidence": 0.95,
                "box": [0, 0, float(w), float(h)],
            }
        ]
        if int(image[300, 300, 0]) == 60:
            detections.append(
                {
                    "class": "discoloration",
                    "confidence": 0.7,
                    "box": [230, 260, 330, 360],
                }
            )
        return detections

    monkeypatch.setattr(pipeline.models, "detect_defects", fake_detect_defects)

    good = make_corn_image()
    bad = make_corn_image(defective=True)
    res = upload([bad, bad, good, good])
    assert res.status_code == 200, res.text
    report = res.json()
    discolor = [d for d in report["defects"] if d["class"] == "discoloration"]
    assert len(discolor) == 1
    assert discolor[0]["merged_count"] == 1
    assert discolor[0]["views_seen"] == [0, 1]
    assert report["defect_coverage"] > 0.0
    assert report["grade"]["grade"] == "ClassI"


def test_inspect_rejects_invalid_upload():
    res = client.post("/inspect", files={"files": ("bad.txt", b"not an image", "text/plain")})
    assert res.status_code == 400


def test_inspect_requires_at_least_one_file():
    res = client.post("/inspect")
    assert res.status_code in (400, 422)


def test_report_images_served_after_inspection():
    res = upload([make_corn_image(), make_corn_image()])
    report = res.json()
    grid = client.get(report["image_url"])
    assert grid.status_code == 200
    assert grid.headers["content-type"] == "image/png"
    for v in report["views"]:
        img = client.get(v["image_url"])
        assert img.status_code == 200
        assert img.headers["content-type"] == "image/png"
