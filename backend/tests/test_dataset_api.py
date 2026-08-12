import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import config
from backend.app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    return TestClient(app)


def _png_bytes(color=(180, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _form(ear_id="auto", variety="yellow_sweet_corn", images=1, labels=0, color=(180, 30, 30)):
    data = {"ear_id": ear_id, "variety": variety}
    files = []
    for i in range(images):
        files.append(("images", (f"photo{i + 1}.png", _png_bytes(color), "image/png")))
    for i in range(labels):
        files.append(("labels", (f"label{i + 1}.txt", b"0 0.5 0.5 0.8 0.8\n", "text/plain")))
    return data, files


def test_upload_ear_writes_tree(client):
    data, files = _form(images=1, labels=1)
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 200, res.text
    assert res.json()["ear_id"] == "ear001"

    raw = config.DATA_DIR / "raw"
    assert (raw / "classifier" / "yellow_sweet_corn" / "ear001_v1.png").exists()
    assert (raw / "detector" / "images" / "ear001_v1.png").exists()
    assert (raw / "detector" / "labels" / "ear001_v1.txt").exists()


def test_upload_auto_ids_increment(client):
    data, files = _form(ear_id="ear007", images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200
    data, files = _form(images=1, color=(30, 30, 200))
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 200, res.text
    assert res.json()["ear_id"] == "ear008"


def test_upload_rejects_duplicate_image(client):
    data, files = _form(ear_id="ear001", images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200
    data, files = _form(images=1)
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 400
    assert "already imported" in res.json()["detail"]


def test_upload_rejects_more_labels_than_images(client):
    data, files = _form(images=1, labels=2)
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 400
    assert "More labels" in res.json()["detail"]


def test_upload_rejects_label_without_corn_ear_box(client):
    data = {"ear_id": "ear001", "variety": "white_corn"}
    files = [
        ("images", ("photo1.png", _png_bytes(), "image/png")),
        ("labels", ("label1.txt", b"2 0.5 0.5 0.1 0.1\n", "text/plain")),
    ]
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 400
    assert "required class 0" in res.json()["detail"]


def test_upload_rejects_unknown_variety(client):
    data, files = _form(variety="purple_corn", images=1)
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 400


def test_ground_truth_upsert_and_autograde(client):
    payload = {
        "ear_id": "ear001",
        "variety": "white_corn",
        "coverage": "0.03",
        "mold_count": 0,
        "insect_count": 0,
        "missing_count": 2,
        "other_count": 0,
    }
    res = client.put("/dataset/ground-truth", json=payload)
    assert res.status_code == 200, res.text
    assert res.json()["row"]["grade"] == "Class I"

    payload["mold_count"] = 1
    res = client.put("/dataset/ground-truth", json=payload)
    assert res.json()["row"]["grade"] == "Reject"

    csv_text = (config.DATA_DIR / "raw" / "ground_truth.csv").read_text()
    assert "ear001,white_corn,0.03,1,0,2,0,Reject" in csv_text


def test_stats_after_upload(client):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    stats = client.get("/dataset/stats").json()
    assert stats["ears"] == 1
    assert stats["detector_images"] == 1
    assert stats["detector_labels"] == 1
    assert stats["missing_labels"] == []
    assert stats["classifier_images"]["yellow_sweet_corn"] == 1

    ears = client.get("/dataset/ears").json()
    assert ears[0]["ear_id"] == "ear001"
    assert ears[0]["views"] == [1]
    assert ears[0]["labels"] == [1]


def test_validate_surfaces_missing_label(client):
    data, files = _form(images=1, labels=0)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    res = client.get("/dataset/validate")
    assert res.status_code == 200
    assert res.json()["ok"] is False
    assert any("missing label" in p for p in res.json()["problems"])


def test_split_requires_confirm(client):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200
    assert client.post("/dataset/split").status_code == 400
    assert client.post("/dataset/split?confirm=true").status_code == 200
