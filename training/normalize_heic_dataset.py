"""Normalize HEIC/HEIF files already present in the raw dataset."""

from __future__ import annotations

import argparse
import errno
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from backend.app import image_io


HEIF_EXTENSIONS = {".heic", ".heif"}
_DESCRIPTOR_RELATIVE_IO = (
    os.name != "nt" and hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")
)


@dataclass(frozen=True)
class MigrationResult:
    """Counts and details produced by one migration run."""

    converted: int
    skipped: int
    failed: int
    failures: tuple[str, ...] = ()


def _scan_candidates_path(input_root: Path) -> tuple[list[Path], list[str]]:
    """Find HEIF files in the two supported raw-dataset locations.

    Classifier files are expected directly in each immediate classifier child
    (normally a variety directory). Detector files are expected directly in
    ``detector/images``. No directory walk follows links.
    """

    candidates: list[Path] = []
    failures: list[str] = []
    if input_root.is_symlink():
        return [], [f"{input_root}: symlinked input root was not followed"]
    scan_roots = (
        (input_root / "classifier", True),
        (input_root / "detector" / "images", False),
    )

    for root, has_class_dirs in scan_roots:
        current = input_root
        symlinked_component: Path | None = None
        for part in root.relative_to(input_root).parts:
            current /= part
            if current.is_symlink():
                symlinked_component = current
                break
        if symlinked_component is not None:
            failures.append(f"{root}: symlinked scan root was not followed")
            continue
        if not root.exists():
            continue
        if not root.is_dir():
            failures.append(f"{root}: expected a directory")
            continue

        try:
            entries = sorted(root.iterdir())
        except OSError as exc:
            failures.append(f"{root}: unable to scan directory: {exc}")
            continue

        if has_class_dirs:
            # The normal layout is classifier/<variety>/<image>.heic. Also
            # accept a file directly below classifier so the glob boundary
            # remains predictable for legacy trees.
            scan_entries: list[Path] = []
            for entry in entries:
                if entry.is_symlink():
                    failures.append(f"{entry}: symlink was not followed")
                    continue
                if entry.is_dir():
                    try:
                        for nested_entry in sorted(entry.iterdir()):
                            if nested_entry.is_symlink():
                                failures.append(f"{nested_entry}: symlink was not followed")
                            else:
                                scan_entries.append(nested_entry)
                    except OSError as exc:
                        failures.append(f"{entry}: unable to scan directory: {exc}")
                else:
                    scan_entries.append(entry)
        else:
            scan_entries = entries

        for entry in scan_entries:
            if entry.is_symlink():
                failures.append(f"{entry}: symlink was not followed")
                continue
            if entry.suffix.lower() not in HEIF_EXTENSIONS:
                continue
            if not entry.is_file():
                failures.append(f"{entry}: expected a regular file")
            else:
                candidates.append(entry)

    return candidates, failures


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_directory_components(
    parent_fd: int, components: tuple[str, ...] | list[str], *, create: bool = False
) -> int:
    """Open a directory below ``parent_fd`` without following any component."""

    current_fd = os.dup(parent_fd)
    try:
        for component in components:
            if component in {"", "."}:
                continue
            try:
                next_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o777, dir_fd=current_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_directory_path(path: Path, *, create: bool = False) -> int:
    """Open a path one component at a time, refusing symlinks."""

    normalized = os.path.normpath(os.fspath(path))
    # macOS exposes these stable system locations as symlinks. Use their
    # kernel-visible targets so user dataset paths under /var and /tmp still
    # get the same no-follow treatment as every other directory component.
    if sys.platform == "darwin":
        for alias, target in (("/var", "/private/var"), ("/tmp", "/private/tmp")):
            if normalized == alias or normalized.startswith(alias + os.path.sep):
                normalized = target + normalized[len(alias) :]
                break
    if os.path.isabs(normalized):
        current_fd = os.open(os.path.sep, _directory_flags())
        components = Path(normalized).parts[1:]
    else:
        current_fd = os.open(".", _directory_flags())
        components = Path(normalized).parts

    try:
        return _open_directory_components(current_fd, components, create=create)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"path component is a symlink: {path}") from exc
        raise
    finally:
        os.close(current_fd)


def _scan_candidates_descriptor(
    input_root: Path, input_root_fd: int
) -> tuple[list[Path], list[str]]:
    """Scan using directory FDs so a replacement cannot redirect the scan."""

    candidates: list[Path] = []
    failures: list[str] = []

    def scan_root(relative_parts: tuple[str, ...], has_class_dirs: bool) -> None:
        root = input_root.joinpath(*relative_parts)
        try:
            root_fd = _open_directory_components(input_root_fd, relative_parts)
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                failures.append(f"{root}: symlinked scan root was not followed")
            elif exc.errno == errno.ENOTDIR:
                failures.append(f"{root}: expected a directory")
            else:
                failures.append(f"{root}: unable to scan directory: {exc}")
            return

        try:
            try:
                entries = sorted(os.listdir(root_fd))
            except OSError as exc:
                failures.append(f"{root}: unable to scan directory: {exc}")
                return

            def add_entry(
                entry_name: str,
                entry_parent_fd: int,
                entry_parts: tuple[str, ...],
                entry_path: Path,
            ) -> None:
                try:
                    entry_stat = os.stat(entry_name, dir_fd=entry_parent_fd, follow_symlinks=False)
                except OSError as exc:
                    failures.append(f"{entry_path}: unable to inspect entry: {exc}")
                    return
                if stat.S_ISLNK(entry_stat.st_mode):
                    failures.append(f"{entry_path}: symlink was not followed")
                else:
                    scan_entries.append((entry_name, entry_parts, entry_path))

            scan_entries: list[tuple[str, tuple[str, ...], Path]] = []
            if has_class_dirs:
                for entry in entries:
                    entry_path = root / entry
                    try:
                        entry_stat = os.stat(entry, dir_fd=root_fd, follow_symlinks=False)
                    except OSError as exc:
                        failures.append(f"{entry_path}: unable to inspect entry: {exc}")
                        continue
                    if stat.S_ISLNK(entry_stat.st_mode):
                        failures.append(f"{entry_path}: symlink was not followed")
                    elif stat.S_ISDIR(entry_stat.st_mode):
                        try:
                            entry_fd = os.open(entry, _directory_flags(), dir_fd=root_fd)
                        except OSError as exc:
                            if exc.errno == errno.ELOOP:
                                failures.append(f"{entry_path}: symlink was not followed")
                            else:
                                failures.append(f"{entry_path}: unable to scan directory: {exc}")
                            continue
                        try:
                            try:
                                nested_entries = sorted(os.listdir(entry_fd))
                            except OSError as exc:
                                failures.append(f"{entry_path}: unable to scan directory: {exc}")
                                continue
                            for nested_entry in nested_entries:
                                add_entry(
                                    nested_entry,
                                    entry_fd,
                                    relative_parts + (entry,),
                                    entry_path / nested_entry,
                                )
                        finally:
                            os.close(entry_fd)
                    else:
                        scan_entries.append((entry, relative_parts, entry_path))
            else:
                for entry in entries:
                    add_entry(entry, root_fd, relative_parts, root / entry)

            for entry_name, entry_parts, entry_path in scan_entries:
                if entry_path.suffix.lower() not in HEIF_EXTENSIONS:
                    continue
                try:
                    entry_parent_fd = _open_directory_components(input_root_fd, entry_parts)
                except OSError as exc:
                    if exc.errno == errno.ELOOP:
                        failures.append(f"{entry_path}: symlink was not followed")
                    else:
                        failures.append(f"{entry_path}: unable to inspect entry: {exc}")
                    continue
                try:
                    entry_stat = os.stat(entry_name, dir_fd=entry_parent_fd, follow_symlinks=False)
                except OSError as exc:
                    failures.append(f"{entry_path}: unable to inspect entry: {exc}")
                else:
                    if not stat.S_ISREG(entry_stat.st_mode):
                        failures.append(f"{entry_path}: expected a regular file")
                    else:
                        candidates.append(input_root.joinpath(*entry_parts, entry_name))
                finally:
                    os.close(entry_parent_fd)
        finally:
            os.close(root_fd)

    scan_root(("classifier",), True)
    scan_root(("detector", "images"), False)
    return candidates, failures


def _scan_candidates(
    input_root: Path, input_root_fd: int | None = None
) -> tuple[list[Path], list[str]]:
    if not _DESCRIPTOR_RELATIVE_IO:
        return _scan_candidates_path(input_root)

    owns_fd = input_root_fd is None
    if owns_fd:
        try:
            input_root_fd = _open_directory_path(input_root)
        except FileNotFoundError:
            return [], []
    assert input_root_fd is not None
    try:
        return _scan_candidates_descriptor(input_root, input_root_fd)
    finally:
        if owns_fd:
            os.close(input_root_fd)


def _validate_no_follow_path(path: Path, final_kind: str) -> None:
    """Validate existing path components without following symlinks."""

    absolute_path = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute_path.anchor)
    components = absolute_path.parts[1:]

    try:
        anchor_stat = os.lstat(current)
    except OSError as exc:
        raise ValueError(f"unable to inspect path component: {current}: {exc}") from exc
    if stat.S_ISLNK(anchor_stat.st_mode):
        raise ValueError(f"path component is a symlink: {current}")
    if not stat.S_ISDIR(anchor_stat.st_mode):
        raise ValueError(f"path component is not a directory: {current}")

    for index, part in enumerate(components):
        current /= part
        try:
            component_stat = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError(f"unable to inspect path component: {current}: {exc}") from exc

        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(f"path component is a symlink: {current}")

        is_final = index == len(components) - 1
        if not is_final or final_kind == "directory":
            if not stat.S_ISDIR(component_stat.st_mode):
                raise ValueError(f"path component is not a directory: {current}")
        elif final_kind == "regular" and not stat.S_ISREG(component_stat.st_mode):
            raise ValueError(f"path component is not a regular file: {current}")


def _check_archive_path(originals_root: Path, relative_path: Path) -> Path:
    """Return a safe archive path and reject symlinked path components."""

    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"unsafe relative dataset path: {relative_path}")

    _validate_no_follow_path(originals_root, final_kind="directory")
    destination = originals_root.joinpath(*relative_path.parts)
    _validate_no_follow_path(destination, final_kind="regular")
    return destination


def _archive_original(
    source: Path, input_root: Path, originals_root: Path, source_bytes: bytes
) -> None:
    relative_path = source.relative_to(input_root)
    destination = _check_archive_path(originals_root, relative_path)
    originals_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink():
        raise ValueError(f"archive destination is a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"archive destination is not a regular file: {destination}")
        if destination.read_bytes() != source_bytes:
            raise ValueError(f"archive destination already contains different bytes: {destination}")
        return

    # Exclusive creation keeps a rerun from overwriting provenance and also
    # avoids following a destination symlink created between the checks.
    try:
        with destination.open("xb") as archive_file:
            archive_file.write(source_bytes)
    except FileExistsError:
        if destination.is_symlink():
            raise ValueError(f"archive destination is a symlink: {destination}")
        if not destination.is_file():
            raise ValueError(f"archive destination is not a regular file: {destination}")
        if destination.read_bytes() != source_bytes:
            raise ValueError(f"archive destination already contains different bytes: {destination}")


def _existing_canonical_jpeg(source: Path) -> Path | None:
    """Find a same-stem JPG without treating a JPEG as the canonical target."""

    target = source.with_suffix(".jpg")
    if target.is_symlink():
        raise ValueError(f"canonical JPG is a symlink: {target}")
    if target.exists():
        return target

    # A case-insensitive filesystem may expose an existing ``.JPG`` under a
    # different spelling. Detect it so it is never overwritten accidentally.
    try:
        siblings = sorted(source.parent.iterdir())
    except OSError as exc:
        raise ValueError(f"unable to inspect canonical JPG siblings: {exc}") from exc
    for sibling in siblings:
        if sibling == source:
            continue
        if sibling.stem == source.stem and sibling.suffix.lower() == ".jpg":
            if sibling.is_symlink():
                raise ValueError(f"canonical JPG is a symlink: {sibling}")
            return sibling
    return None


def _read_regular_file_at(parent_fd: int, name: str, display_path: Path) -> bytes:
    """Read a regular file without following a replaced final symlink."""

    try:
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"symlink was not followed: {display_path}") from exc
        raise

    try:
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"expected a regular file: {display_path}")
        with os.fdopen(file_fd, "rb") as file_handle:
            file_fd = -1
            return file_handle.read()
    finally:
        if file_fd != -1:
            os.close(file_fd)


def _create_exclusive_file_at(parent_fd: int, name: str, contents: bytes) -> None:
    """Create a file below an open directory without following its name."""

    file_fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o666,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(file_fd, "wb") as file_handle:
            file_fd = -1
            file_handle.write(contents)
    finally:
        if file_fd != -1:
            os.close(file_fd)


def _existing_canonical_jpeg_at(
    source_parent_fd: int, source: Path
) -> str | None:
    """Find a same-stem JPG using no-follow descriptor-relative checks."""

    target_name = source.with_suffix(".jpg").name
    target_path = source.with_suffix(".jpg")
    try:
        target_stat = os.stat(target_name, dir_fd=source_parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        target_stat = None
    except OSError:
        raise
    if target_stat is not None:
        if stat.S_ISLNK(target_stat.st_mode):
            raise ValueError(f"canonical JPG is a symlink: {target_path}")
        if not stat.S_ISREG(target_stat.st_mode):
            raise ValueError(f"canonical JPG is not a regular file: {target_path}")
        return target_name

    # A case-insensitive filesystem may expose an existing ``.JPG`` under a
    # different spelling. Detect it so it is never overwritten accidentally.
    try:
        siblings = sorted(os.listdir(source_parent_fd))
    except OSError as exc:
        raise ValueError(f"unable to inspect canonical JPG siblings: {exc}") from exc
    for sibling in siblings:
        if sibling == source.name:
            continue
        sibling_path = source.parent / sibling
        if Path(sibling).stem == source.stem and Path(sibling).suffix.lower() == ".jpg":
            try:
                sibling_stat = os.stat(sibling, dir_fd=source_parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"unable to inspect canonical JPG: {sibling_path}: {exc}") from exc
            if stat.S_ISLNK(sibling_stat.st_mode):
                raise ValueError(f"canonical JPG is a symlink: {sibling_path}")
            if not stat.S_ISREG(sibling_stat.st_mode):
                raise ValueError(f"canonical JPG is not a regular file: {sibling_path}")
            return sibling
    return None


def _archive_original_descriptor(
    source: Path,
    input_root: Path,
    originals_root: Path,
    source_bytes: bytes,
) -> None:
    """Archive bytes using an open, no-follow directory chain."""

    relative_path = source.relative_to(input_root)
    if relative_path.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        raise ValueError(f"unsafe relative dataset path: {relative_path}")

    originals_fd = _open_directory_path(originals_root, create=True)
    archive_parent_fd = -1
    try:
        archive_parent_fd = _open_directory_components(
            originals_fd, relative_path.parent.parts, create=True
        )
        destination = originals_root.joinpath(*relative_path.parts)
        try:
            _create_exclusive_file_at(archive_parent_fd, relative_path.name, source_bytes)
        except FileExistsError:
            # The collision read is also no-follow, so a symlink introduced
            # after the exclusive-create attempt cannot redirect the read.
            existing_bytes = _read_regular_file_at(
                archive_parent_fd, relative_path.name, destination
            )
            if existing_bytes != source_bytes:
                raise ValueError(
                    f"archive destination already contains different bytes: {destination}"
                )
    finally:
        if archive_parent_fd != -1:
            os.close(archive_parent_fd)
        os.close(originals_fd)


def _migrate_one_descriptor(
    source: Path,
    input_root: Path,
    originals_root: Path,
    input_root_fd: int,
) -> str:
    relative_path = source.relative_to(input_root)
    source_parent_fd = _open_directory_components(input_root_fd, relative_path.parent.parts)
    try:
        source_bytes = _read_regular_file_at(source_parent_fd, relative_path.name, source)
        _, canonical_bytes = image_io.normalize_image_bytes(source_bytes)
        canonical_name = _existing_canonical_jpeg_at(source_parent_fd, source)

        if canonical_name is not None:
            canonical = source.with_name(canonical_name)
            try:
                image_io.decode_image_bytes(
                    _read_regular_file_at(source_parent_fd, canonical_name, canonical)
                )
            except Exception as exc:
                raise ValueError(f"existing JPG cannot be decoded ({canonical}): {exc}") from exc
            _archive_original_descriptor(source, input_root, originals_root, source_bytes)
            return "skipped"

        _archive_original_descriptor(source, input_root, originals_root, source_bytes)
        target_name = source.with_suffix(".jpg").name
        target = source.with_suffix(".jpg")
        try:
            _create_exclusive_file_at(source_parent_fd, target_name, canonical_bytes)
        except FileExistsError:
            # Do not overwrite a JPG if another process created it during this
            # run. It is only safe to call this a skip when it decodes cleanly.
            try:
                image_io.decode_image_bytes(
                    _read_regular_file_at(source_parent_fd, target_name, target)
                )
            except Exception as exc:
                raise ValueError(f"existing JPG cannot be decoded ({target}): {exc}") from exc
            return "skipped"
        return "converted"
    finally:
        os.close(source_parent_fd)


def _migrate_one_path(source: Path, input_root: Path, originals_root: Path) -> str:
    source_bytes = source.read_bytes()
    _, canonical_bytes = image_io.normalize_image_bytes(source_bytes)
    canonical = _existing_canonical_jpeg(source)

    if canonical is not None:
        try:
            image_io.decode_image_bytes(canonical.read_bytes())
        except Exception as exc:
            raise ValueError(f"existing JPG cannot be decoded ({canonical}): {exc}") from exc
        _archive_original(source, input_root, originals_root, source_bytes)
        return "skipped"

    _archive_original(source, input_root, originals_root, source_bytes)
    target = source.with_suffix(".jpg")
    try:
        with target.open("xb") as canonical_file:
            canonical_file.write(canonical_bytes)
    except FileExistsError:
        # Do not overwrite a JPG if another process created it during this
        # run. It is only safe to call this a skip when it decodes cleanly.
        if target.is_symlink():
            raise ValueError(f"canonical JPG is a symlink: {target}")
        try:
            image_io.decode_image_bytes(target.read_bytes())
        except Exception as exc:
            raise ValueError(f"existing JPG cannot be decoded ({target}): {exc}") from exc
        return "skipped"
    return "converted"


def _migrate_one(
    source: Path,
    input_root: Path,
    originals_root: Path,
    input_root_fd: int | None = None,
) -> str:
    if _DESCRIPTOR_RELATIVE_IO:
        if input_root_fd is None:
            input_root_fd = _open_directory_path(input_root)
            try:
                return _migrate_one_descriptor(source, input_root, originals_root, input_root_fd)
            finally:
                os.close(input_root_fd)
        return _migrate_one_descriptor(source, input_root, originals_root, input_root_fd)
    return _migrate_one_path(source, input_root, originals_root)


def normalize_heic_dataset(input_root: str | Path, originals_root: str | Path) -> MigrationResult:
    """Convert raw HEIC/HEIF files and archive their untouched source bytes."""

    input_root = Path(input_root)
    originals_root = Path(originals_root)
    if not _DESCRIPTOR_RELATIVE_IO:
        return MigrationResult(
            0,
            0,
            1,
            (
                f"{input_root}: secure descriptor-relative no-follow I/O is "
                "unavailable; migration aborted",
            ),
        )

    input_root_fd: int | None = None
    try:
        if _DESCRIPTOR_RELATIVE_IO:
            try:
                input_root_fd = _open_directory_path(input_root)
            except FileNotFoundError:
                return MigrationResult(0, 0, 0)
            except ValueError as exc:
                return MigrationResult(0, 0, 1, (f"{input_root}: {exc}",))
            except OSError as exc:
                return MigrationResult(0, 0, 1, (f"{input_root}: {exc}",))

        candidates, scan_failures = _scan_candidates(input_root, input_root_fd)
        converted = 0
        skipped = 0
        failures = list(scan_failures)

        for source in candidates:
            try:
                outcome = _migrate_one(source, input_root, originals_root, input_root_fd)
            except Exception as exc:
                failures.append(f"{source}: {exc}")
                continue
            if outcome == "converted":
                converted += 1
            else:
                skipped += 1

        return MigrationResult(converted, skipped, len(failures), tuple(failures))
    finally:
        if input_root_fd is not None:
            os.close(input_root_fd)


# A descriptive alias for callers that use migration terminology.
migrate_heic_dataset = normalize_heic_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize HEIC/HEIF files in the Maisagip raw dataset.")
    parser.add_argument("--input-root", default="data/raw")
    parser.add_argument("--originals-root", default="data/raw/originals")
    args = parser.parse_args(argv)

    result = normalize_heic_dataset(args.input_root, args.originals_root)
    for failure in result.failures:
        print(f"FAILED {failure}")
    print(f"converted: {result.converted}")
    print(f"skipped: {result.skipped}")
    print(f"failed: {result.failed}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
