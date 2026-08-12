from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import config
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


def test_duplicate_defect_across_two_views_merges_to_one():
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