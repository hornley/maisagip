import os
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from backend.app import image_io
import training.normalize_heic_dataset as migration
from training.normalize_heic_dataset import main, normalize_heic_dataset


def _heic_bytes(color: tuple[int, int, int]) -> bytes:
    pillow_heif = pytest.importorskip("pillow_heif", reason="pillow_heif is unavailable")
    source = Image.new("RGB", (8, 6), color=color)
    heif_file = pillow_heif.from_pillow(source)
    output = BytesIO()
    heif_file.save(output)
    return output.getvalue()


def _write_valid_jpg(path: Path, color=(20, 80, 140)) -> bytes:
    image = Image.new("RGB", (8, 6), color=color)
    encoded = image_io.encode_canonical_jpeg(image)
    path.write_bytes(encoded)
    return encoded


def test_normalizes_classifier_and_detector_and_archives_originals(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    classifier_file = input_root / "classifier" / "white_corn" / "ear001_v1.heic"
    detector_file = input_root / "detector" / "images" / "ear002_v3.heif"
    classifier_file.parent.mkdir(parents=True)
    detector_file.parent.mkdir(parents=True)
    classifier_bytes = _heic_bytes((220, 40, 30))
    detector_bytes = _heic_bytes((30, 150, 60))
    classifier_file.write_bytes(classifier_bytes)
    detector_file.write_bytes(detector_bytes)
    label = input_root / "detector" / "labels" / "ear002_v3.txt"
    label.parent.mkdir()
    label.write_text("0 0.5 0.5 0.8 0.8\n", encoding="utf-8")

    originals_root = input_root / "originals"
    result = normalize_heic_dataset(input_root, originals_root)

    assert (result.converted, result.skipped, result.failed) == (2, 0, 0)
    assert (classifier_file.with_suffix(".jpg")).exists()
    assert (detector_file.with_suffix(".jpg")).exists()
    assert (originals_root / "classifier" / "white_corn" / classifier_file.name).read_bytes() == classifier_bytes
    assert (originals_root / "detector" / "images" / detector_file.name).read_bytes() == detector_bytes
    assert label.exists()
    assert not (input_root / "detector" / "labels" / "ear002_v3.jpg").exists()

    with Image.open(classifier_file.with_suffix(".jpg")) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"


def test_valid_existing_jpg_is_skipped_and_original_is_archived(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    source = input_root / "classifier" / "yellow_sweet_corn" / "ear003_v1.heic"
    source.parent.mkdir(parents=True)
    source_bytes = _heic_bytes((90, 100, 110))
    source.write_bytes(source_bytes)
    existing_jpg = source.with_suffix(".jpg")
    existing_bytes = _write_valid_jpg(existing_jpg, color=(1, 2, 3))

    result = normalize_heic_dataset(input_root, input_root / "originals")

    assert (result.converted, result.skipped, result.failed) == (0, 1, 0)
    assert existing_jpg.read_bytes() == existing_bytes
    assert (input_root / "originals" / "classifier" / "yellow_sweet_corn" / source.name).read_bytes() == source_bytes


def test_symlinked_archive_parent_is_rejected_without_writing_external_target(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    source = input_root / "classifier" / "white_corn" / "ear010_v1.heic"
    source.parent.mkdir(parents=True)
    source.write_bytes(_heic_bytes((70, 80, 90)))

    external_target = tmp_path / "external"
    external_target.mkdir()
    symlinked_parent = tmp_path / "archive-parent"
    try:
        symlinked_parent.symlink_to(external_target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    result = normalize_heic_dataset(input_root, symlinked_parent / "originals")

    assert (result.converted, result.skipped, result.failed) == (0, 0, 1)
    assert result.failures
    assert "symlink" in result.failures[0]
    assert not (external_target / "originals").exists()
    assert not source.with_suffix(".jpg").exists()


def test_symlinked_input_root_parent_is_rejected_without_writing_external_target(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    external_parent = tmp_path / "external-parent"
    external_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(external_parent, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    input_root = linked_parent / "raw"
    source = input_root / "classifier" / "white_corn" / "ear011_v1.heic"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source bytes")
    monkeypatch.setattr(
        image_io, "normalize_image_bytes", lambda source_bytes: (None, b"canonical bytes")
    )

    exit_code = main(
        ["--input-root", str(input_root), "--originals-root", str(tmp_path / "originals")]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "FAILED" in output
    assert "symlink" in output
    assert "converted: 0" in output
    assert "skipped: 0" in output
    assert "failed: 1" in output
    assert not source.with_suffix(".jpg").exists()
    assert not (tmp_path / "originals").exists()


def test_symlinked_scan_entries_are_reported_without_being_followed(
    tmp_path: Path, capsys
) -> None:
    input_root = tmp_path / "raw"
    classifier = input_root / "classifier"
    valid_variety = classifier / "white_corn"
    valid_variety.mkdir(parents=True)
    (valid_variety / "notes.txt").write_text("not a candidate", encoding="utf-8")

    linked_variety_target = tmp_path / "linked_variety_target"
    linked_variety_target.mkdir()
    (linked_variety_target / "ear005_v1.heic").write_bytes(b"not a HEIF file")
    (classifier / "linked_variety").symlink_to(linked_variety_target, target_is_directory=True)

    linked_candidate_target = tmp_path / "linked_candidate.heic"
    linked_candidate_target.write_bytes(b"not a HEIF file")
    (valid_variety / "ear006_v1.heic").symlink_to(linked_candidate_target)

    linked_nested_target = tmp_path / "linked_nested_target"
    linked_nested_target.mkdir()
    (linked_nested_target / "ear007_v1.heic").write_bytes(b"not a HEIF file")
    (valid_variety / "linked_nested").symlink_to(linked_nested_target, target_is_directory=True)

    detector_target = tmp_path / "detector_target"
    detector_target.mkdir()
    (detector_target / "ear008_v1.heic").write_bytes(b"not a HEIF file")
    (input_root / "detector").mkdir()
    (input_root / "detector" / "images").symlink_to(detector_target, target_is_directory=True)

    exit_code = main(["--input-root", str(input_root), "--originals-root", str(input_root / "originals")])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "converted: 0" in output
    assert "skipped: 0" in output
    assert "failed: 4" in output
    assert not (linked_variety_target / "ear005_v1.jpg").exists()
    assert not linked_candidate_target.with_suffix(".jpg").exists()
    assert not (linked_nested_target / "ear007_v1.jpg").exists()
    assert not (detector_target / "ear008_v1.jpg").exists()
    assert output.count("FAILED ") == 4


@pytest.mark.skipif(
    os.name == "nt" or not migration._DESCRIPTOR_RELATIVE_IO,
    reason="descriptor-relative no-follow I/O is unavailable on this platform",
)
def test_source_replaced_by_symlink_after_scan_is_not_read_or_written_externally(
    tmp_path: Path, monkeypatch
) -> None:
    input_root = tmp_path / "raw"
    source = input_root / "classifier" / "white_corn" / "ear012_v1.heic"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source bytes")

    external_source = tmp_path / "external.heic"
    external_source.write_bytes(b"external bytes")
    originals_root = input_root / "originals"
    real_scan = migration._scan_candidates

    def scan_then_replace(root: Path, input_root_fd: int | None = None):
        candidates, failures = real_scan(root, input_root_fd)
        source.unlink()
        source.symlink_to(external_source)
        return candidates, failures

    monkeypatch.setattr(migration, "_scan_candidates", scan_then_replace)
    monkeypatch.setattr(
        image_io, "normalize_image_bytes", lambda source_bytes: (None, b"canonical bytes")
    )

    result = normalize_heic_dataset(input_root, originals_root)

    assert (result.converted, result.skipped, result.failed) == (0, 0, 1)
    assert result.failures
    assert "symlink" in result.failures[0]
    assert not (external_source.with_suffix(".jpg")).exists()
    assert not originals_root.exists()


def test_corrupt_heic_fails_even_when_valid_same_stem_jpg_exists(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    source = input_root / "classifier" / "white_corn" / "ear009_v1.heic"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not a HEIF file")
    existing_jpg = source.with_suffix(".jpg")
    existing_bytes = _write_valid_jpg(existing_jpg, color=(4, 5, 6))

    result = normalize_heic_dataset(input_root, input_root / "originals")

    assert (result.converted, result.skipped, result.failed) == (0, 0, 1)
    assert existing_jpg.read_bytes() == existing_bytes
    assert not (input_root / "originals").exists()


def test_corrupt_file_is_reported_and_other_conversions_continue(tmp_path: Path, capsys) -> None:
    input_root = tmp_path / "raw"
    detector = input_root / "detector" / "images"
    detector.mkdir(parents=True)
    good_one = detector / "ear004_v1.heic"
    corrupt = detector / "ear005_v1.heif"
    good_two = detector / "ear006_v1.heic"
    good_one.write_bytes(_heic_bytes((200, 20, 20)))
    corrupt.write_bytes(b"not a HEIF file")
    good_two.write_bytes(_heic_bytes((20, 20, 200)))

    exit_code = main(["--input-root", str(input_root), "--originals-root", str(input_root / "originals")])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "FAILED" in output
    assert "ear005_v1.heif" in output
    assert "converted: 2" in output
    assert "skipped: 0" in output
    assert "failed: 1" in output
    assert good_one.with_suffix(".jpg").exists()
    assert good_two.with_suffix(".jpg").exists()


@pytest.mark.skipif(
    not migration._DESCRIPTOR_RELATIVE_IO,
    reason="absent-input no-op is preserved on supported POSIX only",
)
def test_absent_input_tree_is_a_clean_no_op(tmp_path: Path, capsys) -> None:
    input_root = tmp_path / "missing"

    exit_code = main(["--input-root", str(input_root), "--originals-root", str(tmp_path / "originals")])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output == "converted: 0\nskipped: 0\nfailed: 0\n"


def test_unavailable_descriptor_io_fails_before_scanning_or_writing(
    tmp_path: Path, monkeypatch
) -> None:
    input_root = tmp_path / "raw"
    source = input_root / "classifier" / "white_corn" / "ear013_v1.heic"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source bytes")
    originals_root = tmp_path / "originals"

    monkeypatch.setattr(migration, "_DESCRIPTOR_RELATIVE_IO", False)

    def scan_must_not_run(*args, **kwargs):
        raise AssertionError("migration scanned before secure-I/O preflight")

    monkeypatch.setattr(migration, "_scan_candidates", scan_must_not_run)

    result = normalize_heic_dataset(input_root, originals_root)

    assert (result.converted, result.skipped, result.failed) == (0, 0, 1)
    assert result.failures == (
        f"{input_root}: secure descriptor-relative no-follow I/O is unavailable; migration aborted",
    )
    assert not source.with_suffix(".jpg").exists()
    assert not originals_root.exists()
