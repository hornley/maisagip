"""Materialize classifier images from the archived detector image copy.

The cloud bundle keeps one image copy per detector view.  This utility rebuilds
the classifier directory from that copy using hardlinks where possible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
VARIETIES = ("white_corn", "yellow_sweet_corn")
VIEWS = (1, 2, 3, 4)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EAR_ID = re.compile(r"^ear\d{3}$")


def _load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest JSON: {path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    splits = manifest.get("splits")
    varieties = manifest.get("ear_varieties")
    if not isinstance(splits, dict) or not isinstance(varieties, dict):
        raise ValueError("Manifest must contain object fields 'splits' and 'ear_varieties'")
    return manifest


def _assignments(manifest: dict) -> list[tuple[str, str, str]]:
    assignments: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    ear_varieties = manifest["ear_varieties"]

    for split in SPLITS:
        split_data = manifest["splits"].get(split)
        if not isinstance(split_data, dict) or not isinstance(split_data.get("ears"), list):
            raise ValueError(f"Manifest split {split!r} must contain an ears list")
        for ear in split_data["ears"]:
            if not isinstance(ear, str) or not EAR_ID.fullmatch(ear):
                raise ValueError(f"Invalid ear ID in split {split!r}: {ear!r}")
            if ear in seen:
                raise ValueError(f"Ear appears in multiple splits: {ear}")
            seen.add(ear)
            variety = ear_varieties.get(ear)
            if variety not in VARIETIES:
                raise ValueError(f"Missing or invalid variety for {ear}: {variety!r}")
            assignments.append((split, ear, variety))

    return assignments


def _preflight(
    manifest: dict, data_root: Path
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for split, ear, variety in _assignments(manifest):
        for view in VIEWS:
            stem = f"{ear}_v{view}"
            source_dir = data_root / "detector" / "images" / split
            matches = [
                path for path in source_dir.glob(f"{stem}.*")
                if path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if len(matches) != 1 or not matches[0].is_file() or matches[0].is_symlink():
                raise ValueError(f"Missing or ambiguous detector image: {source_dir / stem}")
            source = matches[0]
            target = data_root / "classifier" / split / variety / source.name
            if target.exists() or target.is_symlink():
                if target.is_symlink() or not target.is_file():
                    raise ValueError(f"Conflicting classifier target: {target}")
                if target.read_bytes() != source.read_bytes():
                    raise ValueError(f"Conflicting classifier target: {target}")
            pairs.append((source, target))
    return pairs


def _materialize_pair(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def materialize(manifest: str | Path = "manifest.json", data_root: str | Path = "data") -> int:
    """Create classifier view files and return the number of materialized files.

    All sources and existing targets are checked before writing. Existing
    byte-identical files are accepted, making reruns idempotent.
    """

    pairs = _preflight(_load_manifest(Path(manifest)), Path(data_root))
    created = 0
    for source, target in pairs:
        if target.exists() or target.is_symlink():
            continue
        _materialize_pair(source, target)
        created += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize Colab classifier images from detector images."
    )
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(f"Materialized {materialize(args.manifest, args.data_root)} classifier images")


if __name__ == "__main__":
    main()
