import json
from zipfile import ZipFile

import pytest

from training.prepare_cloud_bundle import _REQUIRED_FILES, create_bundle


def _project(tmp_path):
    project = tmp_path / "project"
    for relative in _REQUIRED_FILES:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    for variety in ("white_corn", "yellow_sweet_corn"):
        (project / "data/raw/classifier" / variety).mkdir(parents=True)
    (project / "data/raw/detector/images").mkdir(parents=True)
    (project / "data/raw/detector/labels").mkdir(parents=True)
    for variety, ear_numbers in (
        ("yellow_sweet_corn", range(1, 17)),
        ("white_corn", range(32, 48)),
    ):
        for ear_number in ear_numbers:
            for view in range(1, 5):
                name = f"ear{ear_number:03d}_v{view}.jpg"
                content = f"{ear_number}-{view}".encode()
                (project / "data/raw/classifier" / variety / name).write_bytes(content)
                (project / "data/raw/detector/images" / name).write_bytes(content)
                (project / "data/raw/detector/labels" / name.replace(".jpg", ".txt")).write_text(
                    "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                )
    return project


def test_create_bundle_contains_aligned_portable_snapshot(tmp_path):
    archive_path = create_bundle(_project(tmp_path), tmp_path / "joint.zip")
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert all(relative.as_posix() in names for relative in _REQUIRED_FILES)
        assert not any(name.startswith("data/classifier/") for name in names)
        assert "data/detector/images/val/ear001_v1.jpg" in names
        assert "data/detector/data.yaml" in names
        assert "requirements-cloud.txt" in names
        assert "CLOUD_README.md" in names
        assert "manifest.json" in names
        yaml = archive.read("data/detector/data.yaml").decode()
        assert "path: data/detector" in yaml
        assert "nc: 8" in yaml
        assert "shriveled_kernels" in yaml
        readme = archive.read("CLOUD_README.md").decode()
        assert "cannot learn" in readme
        assert "held-out test split" in readme
        assert "python -m training.materialize_cloud_classifier" in readme
        assert "--batch-size 4" in readme
        manifest = json.loads(archive.read("manifest.json"))
        assert "ear001" in manifest["splits"]["val"]["ears"]
        assert "ear013" in manifest["splits"]["val"]["ears"]
        assert {"ear002", "ear011", "ear015"}.issubset(manifest["splits"]["test"]["ears"])
        assert manifest["ear_varieties"]["ear003"] == "yellow_sweet_corn"
        assert manifest["ear_varieties"]["ear032"] == "white_corn"
        for split in ("train", "val", "test"):
            assert manifest["splits"][split]["variety_ear_counts"]["white_corn"] > 0
            assert manifest["splits"][split]["detector_class_counts"]["corn_ear"] == manifest["splits"][split]["image_count"]
            for ear in manifest["splits"][split]["ears"]:
                for view in range(1, 5):
                    name = f"{ear}_v{view}.jpg"
                    assert f"data/detector/images/{split}/{name}" in names


def test_create_bundle_rejects_missing_view_and_label(tmp_path):
    project = _project(tmp_path)
    (project / "data/raw/classifier/white_corn/ear032_v4.jpg").unlink()
    with pytest.raises(ValueError, match="exactly four views"):
        create_bundle(project, tmp_path / "bad.zip")

    project = _project(tmp_path / "second")
    (project / "data/raw/detector/labels/ear001_v1.txt").unlink()
    with pytest.raises(ValueError, match="Missing detector label"):
        create_bundle(project, tmp_path / "bad-label.zip")


def test_create_bundle_rejects_unpaired_detector_view(tmp_path):
    project = _project(tmp_path)
    (project / "data/raw/detector/images/ear001_v1.jpg").unlink()
    with pytest.raises(ValueError, match="exactly four views"):
        create_bundle(project, tmp_path / "bad-pair.zip")
