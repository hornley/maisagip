"""Rewrite the dataset root in a YOLO dataset YAML file.

This is useful after moving the prepared dataset from a local machine to
Colab, Kaggle, or another training host where the absolute path has changed.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


_PATH_LINE = re.compile(r"^(\s*path\s*:\s*)(.*?)(\s*)$")


def rewrite_data_yaml(
    data_yaml: str | Path,
    dataset_root: str | Path,
    *,
    output: str | Path | None = None,
    in_place: bool = False,
    relative_to: str | Path | None = None,
) -> Path:
    """Write a copy of ``data_yaml`` with its ``path`` replaced.

    ``dataset_root`` must contain the ``images`` and ``labels`` directories.
    Pass ``relative_to`` to write a portable relative path instead of an
    absolute path.
    """
    source = Path(data_yaml).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {source}")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not (root / "images").is_dir() or not (root / "labels").is_dir():
        raise ValueError(
            f"Dataset root must contain images/ and labels/: {root}"
        )
    if in_place and output is not None:
        raise ValueError("Use either --in-place or --output, not both")

    replacement = root
    if relative_to is not None:
        base = Path(relative_to).expanduser().resolve()
        replacement = Path(os.path.relpath(root, base))
    replacement_text = replacement.as_posix()

    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    path_indexes = []
    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        if _PATH_LINE.match(body):
            path_indexes.append(index)

    if len(path_indexes) != 1:
        raise ValueError(
            f"Expected exactly one top-level path: entry in {source}; "
            f"found {len(path_indexes)}"
        )

    index = path_indexes[0]
    newline = "\n" if lines[index].endswith("\n") else ""
    lines[index] = f"path: {replacement_text}{newline}"

    if in_place:
        destination = source
    elif output is not None:
        destination = Path(output).expanduser().resolve()
    else:
        destination = source.with_name(f"{source.stem}.cloud{source.suffix}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(lines), encoding="utf-8")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite the path: entry in a YOLO dataset YAML file."
    )
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--relative-to",
        type=Path,
        help="Write a relative dataset path from this directory.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    destination = rewrite_data_yaml(
        args.data_yaml,
        args.dataset_root,
        output=args.output,
        in_place=args.in_place,
        relative_to=args.relative_to,
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
