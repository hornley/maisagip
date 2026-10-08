import json
from pathlib import Path

import pytest

import training.materialize_cloud_classifier as materializer
from training.materialize_cloud_classifier import materialize


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    manifest = tmp_path / "manifest.json"
    payload = {
        "ear_varieties": {"ear001": "white_corn", "ear002": "yellow_sweet_corn"},
        "splits": {
            "train": {"ears": ["ear001"]},
            "val": {"ears": ["ear002"]},
            "test": {"ears": []},
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    for split, ears in (("train", ["ear001"]), ("val", ["ear002"])):
        source_dir = data / "detector" / "images" / split
        source_dir.mkdir(parents=True)
        for ear in ears:
            for view in range(1, 5):
                (source_dir / f"{ear}_v{view}.jpg").write_bytes(f"{ear}-{view}".encode())
    return manifest, data


def test_materializes_all_views_into_manifest_variety_directories(tmp_path):
    manifest, data = _fixture(tmp_path)

    assert materialize(manifest, data) == 8
    for split, variety, ear in (
        ("train", "white_corn", "ear001"),
        ("val", "yellow_sweet_corn", "ear002"),
    ):
        for view in range(1, 5):
            source = data / "detector" / "images" / split / f"{ear}_v{view}.jpg"
            target = data / "classifier" / split / variety / source.name
            assert target.read_bytes() == source.read_bytes()
            assert target.stat().st_ino == source.stat().st_ino


def test_materialize_is_idempotent(tmp_path):
    manifest, data = _fixture(tmp_path)

    assert materialize(manifest, data) == 8
    assert materialize(manifest, data) == 0


def test_missing_view_fails_before_creating_any_classifier_output(tmp_path):
    manifest, data = _fixture(tmp_path)
    (data / "detector" / "images" / "val" / "ear002_v4.jpg").unlink()

    with pytest.raises(ValueError, match="detector image"):
        materialize(manifest, data)
    assert not (data / "classifier").exists()


def test_materialize_preserves_non_jpg_image_extension(tmp_path):
    manifest, data = _fixture(tmp_path)
    source = data / "detector" / "images" / "train" / "ear001_v1.jpg"
    png = source.with_suffix(".png")
    source.rename(png)

    assert materialize(manifest, data) == 8
    assert (data / "classifier" / "train" / "white_corn" / png.name).read_bytes() == png.read_bytes()


def test_conflicting_target_is_rejected_before_creating_other_targets(tmp_path):
    manifest, data = _fixture(tmp_path)
    target = data / "classifier" / "train" / "white_corn" / "ear001_v1.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different")

    with pytest.raises(ValueError, match="Conflicting classifier target"):
        materialize(manifest, data)
    assert not (target.parent / "ear001_v2.jpg").exists()


def test_materialize_falls_back_to_copy_when_hardlinks_are_unavailable(tmp_path, monkeypatch):
    manifest, data = _fixture(tmp_path)

    def no_hardlinks(*args):
        raise OSError("hardlinks unavailable")

    monkeypatch.setattr(materializer.os, "link", no_hardlinks)
    assert materialize(manifest, data) == 8
    source = data / "detector" / "images" / "train" / "ear001_v1.jpg"
    target = data / "classifier" / "train" / "white_corn" / source.name
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino
