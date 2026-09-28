import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import config
from backend.app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "ORIGINALS_DIR", tmp_path / "data" / "raw" / "originals")
    return TestClient(app)


def _png_bytes(color=(180, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _nontrivial_png_bytes():
    image = Image.new("RGB", (64, 64))
    pixels = image.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (
                (x * 17 + y * 3) % 256,
                (x * 5 + y * 19) % 256,
                (x * 11 + y * 7) % 256,
            )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _heic_bytes(color=(180, 30, 30)):
    pillow_heif = pytest.importorskip("pillow_heif", reason="pillow_heif is unavailable")
    source = Image.new("RGB", (64, 64), color=color)
    heif_file = pillow_heif.from_pillow(source)
    buf = io.BytesIO()
    heif_file.save(buf)
    return buf.getvalue()


def _form(ear_id="auto", variety="yellow_sweet_corn", images=1, labels=0, color=(180, 30, 30)):
    data = {"ear_id": ear_id, "variety": variety}
    files = []
    for i in range(images):
        files.append(("images", (f"photo{i + 1}.png", _png_bytes(color), "image/png")))
    for i in range(labels):
        files.append(("labels", (f"label{i + 1}.txt", b"0 0.5 0.5 0.8 0.8\n", "text/plain")))
    return data, files


def _multi_image_form(ear_id="auto", images=4):
    data = {"ear_id": ear_id, "variety": "yellow_sweet_corn"}
    files = []
    for i in range(images):
        files.append(("images", (f"photo{i + 1}.png", _png_bytes(color=(30 + i * 40, 30, 200)), "image/png")))
    return data, files


def _bulk_files(groups=("ear001", "ear002")):
    files = []
    for group_index, source_ear in enumerate(groups):
        for view in range(1, 5):
            color = (30 + group_index * 60, 30 + view * 35, 200 - view * 20)
            files.append(
                (
                    "images",
                    (f"{source_ear}_v{view}.png", _png_bytes(color=color), "image/png"),
                )
            )
    return files


def test_upload_ear_writes_tree(client):
    data, files = _form(images=1, labels=1)
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 200, res.text
    assert res.json()["ear_id"] == "ear001"

    raw = config.DATA_DIR / "raw"
    assert (raw / "originals" / "ear001" / "photo1.png").exists()
    assert (raw / "classifier" / "yellow_sweet_corn" / "ear001_v1.jpg").exists()
    assert (raw / "detector" / "images" / "ear001_v1.jpg").exists()
    assert (raw / "detector" / "labels" / "ear001_v1.txt").exists()

    with Image.open(raw / "detector" / "images" / "ear001_v1.jpg") as canonical:
        assert canonical.format == "JPEG"
        assert canonical.mode == "RGB"


def test_upload_auto_ids_increment(client):
    data, files = _form(ear_id="ear007", images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200
    data, files = _form(images=1, color=(30, 30, 200))
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 200, res.text
    assert res.json()["ear_id"] == "ear008"


def test_upload_requires_three_digit_ear_id(client):
    data, files = _form(ear_id="ear1", images=1)
    res = client.post("/dataset/ears", data=data, files=files)
    assert res.status_code == 400
    assert "expected 'earNNN'" in res.json()["detail"]


def test_bulk_upload_groups_views_and_assigns_new_dataset_ids(client):
    data, files = _form(ear_id="ear007", images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    files = _bulk_files(("ear001", "ear011"))
    res = client.post(
        "/dataset/bulk-ears",
        data={"variety": "yellow_sweet_corn"},
        files=files,
    )

    assert res.status_code == 200, res.text
    assert res.json() == {
        "ears_imported": 2,
        "views_imported": 8,
        "ear_ids": ["ear008", "ear009"],
    }
    raw = config.DATA_DIR / "raw"
    for ear_id in ("ear008", "ear009"):
        for view in range(1, 5):
            assert (raw / "classifier" / "yellow_sweet_corn" / f"{ear_id}_v{view}.jpg").exists()
            assert (raw / "detector" / "images" / f"{ear_id}_v{view}.jpg").exists()
    assert (raw / "originals" / "ear008" / "ear001_v1.png").exists()
    assert (raw / "originals" / "ear009" / "ear011_v4.png").exists()


def test_bulk_upload_rejects_incomplete_group_atomically(client):
    files = _bulk_files(("ear001",))[:-1]
    res = client.post(
        "/dataset/bulk-ears",
        data={"variety": "white_corn"},
        files=files,
    )

    assert res.status_code == 400
    assert "missing v4" in res.json()["detail"]
    assert not config.DATA_DIR.exists()


def test_bulk_upload_requires_three_digit_source_group_ids(client):
    files = _bulk_files(("ear1",))
    res = client.post(
        "/dataset/bulk-ears",
        data={"variety": "white_corn"},
        files=files,
    )

    assert res.status_code == 400
    assert "ear1_v1.png" in res.json()["detail"]
    assert not config.DATA_DIR.exists()


def test_bulk_upload_reserves_archive_only_ids(client):
    archive = config.ORIGINALS_DIR / "ear001"
    archive.mkdir(parents=True)

    res = client.post(
        "/dataset/bulk-ears",
        data={"variety": "white_corn"},
        files=_bulk_files(("ear099",)),
    )

    assert res.status_code == 200, res.text
    assert res.json()["ear_ids"] == ["ear002"]


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


def test_heic_upload_archives_original_and_writes_canonical_jpegs(client):
    heic = _heic_bytes()
    data = {"ear_id": "auto", "variety": "yellow_sweet_corn"}
    files = [("images", ("photo.heic", heic, "image/heic"))]

    res = client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 200, res.text
    raw = config.DATA_DIR / "raw"
    original = raw / "originals" / "ear001" / "photo.heic"
    classifier = raw / "classifier" / "yellow_sweet_corn" / "ear001_v1.jpg"
    detector = raw / "detector" / "images" / "ear001_v1.jpg"
    assert original.read_bytes() == heic
    assert classifier.read_bytes() == detector.read_bytes()
    with Image.open(classifier) as canonical:
        assert canonical.format == "JPEG"
        assert canonical.mode == "RGB"


def test_png_then_equivalent_heic_is_rejected_as_duplicate(client):
    data, files = _form(images=1, color=(0, 0, 0))
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    heic = _heic_bytes(color=(0, 0, 0))
    duplicate = client.post(
        "/dataset/ears",
        data={"ear_id": "auto", "variety": "yellow_sweet_corn"},
        files=[("images", ("equivalent.heic", heic, "image/heic"))],
    )

    assert duplicate.status_code == 400
    assert "already imported" in duplicate.json()["detail"]


def test_nontrivial_legacy_jpeg_duplicate_uses_normalized_upload_pixels(client):
    legacy_path = config.DATA_DIR / "raw" / "classifier" / "yellow_sweet_corn" / "legacy.jpg"
    legacy_path.parent.mkdir(parents=True)
    source = Image.open(io.BytesIO(_nontrivial_png_bytes())).convert("RGB")
    legacy = io.BytesIO()
    source.save(legacy, format="JPEG", quality=20, comment=b"legacy metadata")
    legacy_path.write_bytes(legacy.getvalue())

    with Image.open(io.BytesIO(legacy.getvalue())) as decoded_legacy:
        upload = io.BytesIO()
        decoded_legacy.convert("RGB").save(upload, format="PNG")

    data = {"ear_id": "auto", "variety": "yellow_sweet_corn"}
    files = [("images", ("equivalent.png", upload.getvalue(), "image/png"))]
    res = client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 400
    assert "already imported" in res.json()["detail"]


def test_repeated_nontrivial_image_is_rejected_as_duplicate(client):
    image = _nontrivial_png_bytes()
    files = [("images", ("photo.png", image, "image/png"))]
    assert client.post(
        "/dataset/ears",
        data={"ear_id": "ear001", "variety": "yellow_sweet_corn"},
        files=files,
    ).status_code == 200

    duplicate = client.post(
        "/dataset/ears",
        data={"ear_id": "auto", "variety": "yellow_sweet_corn"},
        files=[("images", ("repeat.png", image, "image/png"))],
    )

    assert duplicate.status_code == 400
    assert "already imported" in duplicate.json()["detail"]


def test_invalid_heic_label_leaves_no_files(client):
    raw = config.DATA_DIR / "raw"
    assert not config.DATA_DIR.exists()

    data = {"ear_id": "ear001", "variety": "yellow_sweet_corn"}
    files = [
        ("images", ("photo.heic", _heic_bytes(), "image/heic")),
        ("labels", ("label1.txt", b"2 0.5 0.5 0.1 0.1\n", "text/plain")),
    ]

    res = client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 400
    for directory in (
        raw,
        raw / "classifier",
        raw / "detector",
        raw / "originals",
    ):
        assert not directory.exists()
    assert not (raw / "originals" / "ear001").exists()


def test_upload_rolls_back_all_new_files_when_commit_fails(client, monkeypatch):
    data, files = _form(ear_id="ear001", images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    raw = config.DATA_DIR / "raw"
    files_before = {path.relative_to(raw) for path in raw.rglob("*") if path.is_file()}
    real_replace = Path.replace
    replace_calls = 0

    def fail_on_second_replace(path, target):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated archive commit failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_on_second_replace)
    failed_client = TestClient(app, raise_server_exceptions=False)
    data, files = _form(images=1, color=(30, 30, 200))
    res = failed_client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 500
    files_after = {path.relative_to(raw) for path in raw.rglob("*") if path.is_file()}
    assert files_after == files_before


def test_upload_rejects_symlinked_archive_root(client, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    originals = config.ORIGINALS_DIR
    originals.parent.mkdir(parents=True)
    try:
        originals.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create directory symlinks")

    data, files = _form(images=1)
    res = client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 400
    assert list(outside.iterdir()) == []


def test_upload_rejects_symlinked_archive_ear_directory(client, tmp_path):
    originals = config.ORIGINALS_DIR
    originals.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (originals / "ear001").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create directory symlinks")

    data, files = _form(ear_id="ear001", images=1)
    res = client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 400
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("classifier") / "yellow_sweet_corn",
        Path("detector") / "images",
        Path("detector") / "labels",
    ],
)
def test_upload_rejects_symlinked_dataset_directory(client, tmp_path, relative_path):
    raw = config.DATA_DIR / "raw"
    target = raw / relative_path
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"keep")
    try:
        target.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create directory symlinks")

    data, files = _form(images=1)
    res = client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 400
    assert target.is_symlink()
    assert sentinel.read_bytes() == b"keep"
    assert list(outside.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("classifier") / "yellow_sweet_corn",
        Path("detector") / "images",
        Path("detector") / "labels",
    ],
)
def test_delete_rejects_symlinked_dataset_directory_without_touching_target(
    client, tmp_path, relative_path
):
    raw = config.DATA_DIR / "raw"
    target = raw / relative_path
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "ear001_v1.jpg"
    sentinel.write_bytes(b"keep")
    try:
        target.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create directory symlinks")

    res = client.delete("/dataset/ears/ear001")

    assert res.status_code == 400
    assert target.is_symlink()
    assert sentinel.read_bytes() == b"keep"


def test_delete_ear_removes_only_that_ears_archive(client):
    data, files = _form(ear_id="ear001", images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    raw = config.DATA_DIR / "raw"
    classifier = raw / "classifier" / "yellow_sweet_corn"
    detector = raw / "detector" / "images"
    (classifier / "ear001_v2.jpg").mkdir()
    (detector / "ear001_v2.jpg").mkdir()
    (classifier / "ear001_v99.jpg").write_bytes(b"auxiliary")
    (detector / "ear001_v1.jpeg").write_bytes(b"auxiliary")

    unrelated = config.ORIGINALS_DIR / "ear0010" / "keep.png"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"keep")

    res = client.delete("/dataset/ears/ear001")

    assert res.status_code == 200, res.text
    assert not (raw / "originals" / "ear001").exists()
    assert unrelated.exists()
    assert not (raw / "classifier" / "yellow_sweet_corn" / "ear001_v1.jpg").exists()
    assert not (raw / "detector" / "images" / "ear001_v1.jpg").exists()
    assert (classifier / "ear001_v2.jpg").is_dir()
    assert (detector / "ear001_v2.jpg").is_dir()
    assert (classifier / "ear001_v99.jpg").exists()
    assert (detector / "ear001_v1.jpeg").exists()
    assert not (raw / "detector" / "labels" / "ear001_v1.txt").exists()


def test_delete_ear_ignores_label_directory_without_partial_deletion(client):
    data, files = _form(ear_id="ear001", images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    raw = config.DATA_DIR / "raw"
    label_target = raw / "detector" / "labels" / "ear001_v1.txt"
    label_target.mkdir()

    res = client.delete("/dataset/ears/ear001")

    assert res.status_code == 200, res.text
    assert res.json()["removed_labels"] == 0
    assert label_target.is_dir()
    assert not (raw / "detector" / "images" / "ear001_v1.jpg").exists()


def test_delete_ear_ignores_label_symlink_without_partial_deletion(client, tmp_path):
    data, files = _form(ear_id="ear001", images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    raw = config.DATA_DIR / "raw"
    label_target = raw / "detector" / "labels" / "ear001_v1.txt"
    outside = tmp_path / "outside-label.txt"
    outside.write_text("0 0.5 0.5 0.8 0.8\n")
    try:
        label_target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create file symlinks")

    res = client.delete("/dataset/ears/ear001")

    assert res.status_code == 200, res.text
    assert res.json()["removed_labels"] == 0
    assert label_target.is_symlink()
    assert outside.exists()
    assert not (raw / "detector" / "images" / "ear001_v1.jpg").exists()


def test_original_archive_uses_safe_noncolliding_basenames(client):
    data = {"ear_id": "ear001", "variety": "yellow_sweet_corn"}
    files = [
        ("images", ("../photo.png", _png_bytes((180, 30, 30)), "image/png")),
        ("images", (r"..\photo.png", _png_bytes((30, 30, 180)), "image/png")),
    ]

    res = client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 200, res.text
    archive = config.ORIGINALS_DIR / "ear001"
    assert (archive / "photo.png").exists()
    assert (archive / "photo-2.png").exists()
    assert not (config.DATA_DIR / "raw" / "photo.png").exists()


def test_original_archive_casefolds_noncolliding_basenames(client):
    data = {"ear_id": "ear001", "variety": "yellow_sweet_corn"}
    files = [
        ("images", ("photo.png", _png_bytes((180, 30, 30)), "image/png")),
        ("images", ("PHOTO.PNG", _png_bytes((30, 30, 180)), "image/png")),
    ]

    res = client.post("/dataset/ears", data=data, files=files)

    assert res.status_code == 200, res.text
    archive = config.ORIGINALS_DIR / "ear001"
    assert (archive / "photo.png").read_bytes() == _png_bytes((180, 30, 30))
    assert (archive / "PHOTO-2.PNG").read_bytes() == _png_bytes((30, 30, 180))


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


def test_attach_labels_bulk(client):
    data, files = _multi_image_form(images=4)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    labels = [("labels", (f"l{i}.txt", b"0 0.5 0.5 0.8 0.8\n", "text/plain")) for i in range(1, 5)]
    views = [("views", (None, str(v))) for v in (1, 2, 3, 4)]
    res = client.post("/dataset/ears/ear001/labels/bulk", files=[*views, *labels])
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["attached"] == 4
    assert set(body["written"]) == {"ear001_v1.txt", "ear001_v2.txt", "ear001_v3.txt", "ear001_v4.txt"}

    raw = config.DATA_DIR / "raw"
    for v in range(1, 5):
        assert (raw / "detector" / "labels" / f"ear001_v{v}.txt").exists()


def test_attach_labels_bulk_rejects_invalid_view(client):
    data, files = _multi_image_form(images=2)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    labels = [("labels", (f"l{i}.txt", b"0 0.5 0.5 0.8 0.8\n", "text/plain")) for i in (1, 2)]
    views = [("views", (None, str(v))) for v in (1, 5)]
    res = client.post("/dataset/ears/ear001/labels/bulk", files=[*views, *labels])
    assert res.status_code == 400
    assert "must be between 1 and 4" in res.json()["detail"]


def test_attach_labels_bulk_rejects_duplicate_view(client):
    data, files = _multi_image_form(images=2)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    labels = [
        ("labels", ("first.txt", b"0 0.5 0.5 0.8 0.8\n", "text/plain")),
        ("labels", ("second.txt", b"0 0.5 0.5 0.8 0.8\n", "text/plain")),
    ]
    views = [("views", (None, "1")), ("views", (None, "1"))]
    res = client.post("/dataset/ears/ear001/labels/bulk", files=[*views, *labels])

    assert res.status_code == 400
    assert "Duplicate view" in res.json()["detail"]


def test_attach_labels_bulk_is_atomic_on_bad_label(client):
    data, files = _multi_image_form(images=2)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    labels = [
        ("labels", ("good.txt", b"0 0.5 0.5 0.8 0.8\n", "text/plain")),
        ("labels", ("bad.txt", b"corn_ear\n", "text/plain")),
    ]
    views = [("views", (None, str(v))) for v in (1, 2)]
    res = client.post("/dataset/ears/ear001/labels/bulk", files=[*views, *labels])
    assert res.status_code == 400
    assert "expected 5 fields" in res.json()["detail"]

    raw = config.DATA_DIR / "raw"
    assert not (raw / "detector" / "labels" / "ear001_v1.txt").exists()
    assert not (raw / "detector" / "labels" / "ear001_v2.txt").exists()


def test_replace_labels_bulk_overwrites_existing_label(client):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200
    replacement = b"0 0.25 0.25 0.4 0.4\n"

    res = client.put(
        "/dataset/ears/ear001/labels/bulk",
        files=[("views", (None, "1")), ("labels", ("replacement.txt", replacement, "text/plain"))],
    )

    assert res.status_code == 200, res.text
    assert (config.DATA_DIR / "raw" / "detector" / "labels" / "ear001_v1.txt").read_bytes() == replacement


def test_put_label_creates_label_for_existing_view(client):
    data, files = _form(images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    content = b"0 0.5 0.5 0.8 0.8\n"
    res = client.put(
        "/dataset/ears/ear001/labels/1",
        files={"label": ("annotation.txt", content, "text/plain")},
    )

    assert res.status_code == 200, res.text
    assert res.json() == {"ear_id": "ear001", "view": 1, "written": "ear001_v1.txt"}
    assert (config.DATA_DIR / "raw" / "detector" / "labels" / "ear001_v1.txt").read_bytes() == content


def test_put_label_overwrites_existing_label(client):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200
    target = config.DATA_DIR / "raw" / "detector" / "labels" / "ear001_v1.txt"
    replacement = b"0 0.25 0.25 0.4 0.4\n"

    res = client.put(
        "/dataset/ears/ear001/labels/1",
        files={"label": ("annotation.txt", replacement, "text/plain")},
    )

    assert res.status_code == 200, res.text
    assert target.read_bytes() == replacement


def test_put_label_rejects_invalid_label_without_overwriting(client):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200
    target = config.DATA_DIR / "raw" / "detector" / "labels" / "ear001_v1.txt"
    original = target.read_bytes()

    res = client.put(
        "/dataset/ears/ear001/labels/1",
        files={"label": ("annotation.txt", b"not a YOLO line\n", "text/plain")},
    )

    assert res.status_code == 400
    assert "expected 5 fields" in res.json()["detail"]
    assert target.read_bytes() == original


def test_put_label_rejects_empty_label(client):
    data, files = _form(images=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    res = client.put(
        "/dataset/ears/ear001/labels/1",
        files={"label": ("annotation.txt", b"", "text/plain")},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "Empty label file."


def test_put_label_rejects_missing_image(client):
    res = client.put(
        "/dataset/ears/ear001/labels/1",
        files={"label": ("annotation.txt", b"0 0.5 0.5 0.8 0.8\n", "text/plain")},
    )

    assert res.status_code == 404
    assert "No image" in res.json()["detail"]


def test_split_requires_confirm(client):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200
    assert client.post("/dataset/split").status_code == 400
    assert client.post("/dataset/split?confirm=true").status_code == 200


def test_split_rejects_symlinked_output_root_without_external_write(client, tmp_path):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"keep")
    output_root = config.DATA_DIR / "classifier"
    try:
        output_root.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create directory symlinks")

    res = client.post("/dataset/split?confirm=true")

    assert res.status_code == 400
    assert output_root.is_symlink()
    assert sentinel.read_bytes() == b"keep"
    assert list(outside.iterdir()) == [sentinel]


def test_split_rejects_symlinked_destination_file_without_external_write(client, tmp_path):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"keep")
    destination = (
        config.DATA_DIR / "classifier" / "test" / "yellow_sweet_corn" / "ear001_v1.jpg"
    )
    destination.parent.mkdir(parents=True)
    try:
        destination.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create file symlinks")

    res = client.post("/dataset/split?confirm=true")

    assert res.status_code == 400
    assert destination.is_symlink()
    assert outside.read_bytes() == b"keep"


def test_split_rejects_symlinked_raw_detector_image_without_external_write(client, tmp_path):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    raw_image = config.DATA_DIR / "raw" / "detector" / "images" / "ear001_v1.jpg"
    outside = tmp_path / "outside-image.jpg"
    outside.write_bytes(b"keep")
    raw_image.unlink()
    try:
        raw_image.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create file symlinks")

    res = client.post("/dataset/split?confirm=true")

    assert res.status_code == 400
    assert raw_image.is_symlink()
    assert outside.read_bytes() == b"keep"
    assert not (config.DATA_DIR / "classifier").exists()
    assert not (config.DATA_DIR / "detector").exists()


def test_split_rejects_symlinked_raw_detector_label_without_external_write(client, tmp_path):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    raw_label = config.DATA_DIR / "raw" / "detector" / "labels" / "ear001_v1.txt"
    outside = tmp_path / "outside-label.txt"
    outside.write_text("0 0.5 0.5 0.8 0.8\n")
    raw_label.unlink()
    try:
        raw_label.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create file symlinks")

    res = client.post("/dataset/split?confirm=true")

    assert res.status_code == 400
    assert raw_label.is_symlink()
    assert outside.read_text() == "0 0.5 0.5 0.8 0.8\n"
    assert not (config.DATA_DIR / "classifier").exists()
    assert not (config.DATA_DIR / "detector").exists()


def test_split_rejects_symlinked_raw_classifier_source_without_external_write(client, tmp_path):
    data, files = _form(images=1, labels=1)
    assert client.post("/dataset/ears", data=data, files=files).status_code == 200

    raw_image = (
        config.DATA_DIR
        / "raw"
        / "classifier"
        / "yellow_sweet_corn"
        / "ear001_v1.jpg"
    )
    outside = tmp_path / "outside-classifier-image.jpg"
    outside.write_bytes(b"keep")
    raw_image.unlink()
    try:
        raw_image.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create file symlinks")

    res = client.post("/dataset/split?confirm=true")

    assert res.status_code == 400
    assert raw_image.is_symlink()
    assert outside.read_bytes() == b"keep"
    assert not (config.DATA_DIR / "classifier").exists()
    assert not (config.DATA_DIR / "detector").exists()
