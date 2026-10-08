import hashlib
import re
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import config
from . import image_io
from training.validate_dataset import check_label_lines
from training.validate_dataset import ear_id as parse_ear_id
from training.validate_dataset import view_number as parse_view_number
from training.make_dataset_split import split_classifier, split_detector

router = APIRouter(prefix="/dataset", tags=["dataset"])

IMAGE_EXTS = image_io.SUPPORTED_IMAGE_EXTS

GRADE_EXTRA = "Extra Class"
GRADE_I = "Class I"
GRADE_II = "Class II"
GRADE_REJECT = "Reject"
BULK_IMAGE_RE = re.compile(r"^(ear\d{3})_v([1-4])\.(heic|heif|jpg|jpeg|png)$", re.IGNORECASE)


def _raw_dir() -> Path:
    return config.DATA_DIR / "raw"


def _classifier_raw() -> Path:
    return _raw_dir() / "classifier"


def _detector_raw() -> Path:
    return _raw_dir() / "detector"


def _detector_images() -> Path:
    return _detector_raw() / "images"


def _detector_labels() -> Path:
    return _detector_raw() / "labels"


def _originals_dir() -> Path:
    return config.ORIGINALS_DIR


def _validate_real_dir(path: Path) -> Path:
    """Validate every existing component of a directory path without following links."""
    path = Path(path)
    current = Path(path.anchor) if path.is_absolute() else Path()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise HTTPException(status_code=400, detail=f"Dataset directory must not be a symlink: {current}.")
        if current.exists() and not current.is_dir():
            raise HTTPException(status_code=400, detail=f"Dataset path must be a real directory: {current}.")
    return path


def _ensure_real_dir(path: Path) -> Path:
    """Create a directory one component at a time, rejecting symlinks."""
    path = _validate_real_dir(path)
    current = Path(path.anchor) if path.is_absolute() else Path()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise HTTPException(status_code=400, detail=f"Dataset directory must not be a symlink: {current}.")
        if not current.exists():
            try:
                current.mkdir()
            except FileExistsError:
                pass
        if current.is_symlink() or not current.is_dir():
            raise HTTPException(status_code=400, detail=f"Dataset path must be a real directory: {current}.")
    return path


def _archive_dir(ear_id_value: str, create: bool) -> Path | None:
    """Return an archive ear directory without traversing symlinked paths."""
    originals_dir = _originals_dir()
    if originals_dir.is_symlink():
        raise HTTPException(status_code=400, detail="Archive root must not be a symlink.")
    if not originals_dir.exists():
        if not create:
            return None
        _ensure_real_dir(originals_dir)
    _validate_real_dir(originals_dir)

    archive_dir = originals_dir / ear_id_value
    if archive_dir.is_symlink():
        raise HTTPException(status_code=400, detail=f"Archive directory for {ear_id_value} must not be a symlink.")
    if not archive_dir.exists():
        if not create:
            return archive_dir
        _ensure_real_dir(archive_dir)
    if archive_dir.is_symlink() or not archive_dir.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Archive directory for {ear_id_value} must be a real directory."
        )
    return archive_dir


def _gt_csv() -> Path:
    return _raw_dir() / "ground_truth.csv"


def _dataset_dirs() -> list[Path]:
    return [
        _raw_dir(),
        _classifier_raw(),
        *(_classifier_raw() / variety for variety in config.VARIETY_CLASSES),
        _detector_raw(),
        _detector_images(),
        _detector_labels(),
        _originals_dir(),
    ]


def _validate_dirs() -> None:
    """Validate dataset directories without creating missing components."""
    for directory in _dataset_dirs():
        _validate_real_dir(directory)


def _ensure_dirs():
    directories = _dataset_dirs()
    # Preflight every path before creating any missing directory. This keeps a
    # later symlink rejection from occurring after dataset writes begin.
    _validate_dirs()
    for directory in directories:
        _ensure_real_dir(directory)


def _split_dirs() -> list[Path]:
    classifier_root = config.DATA_DIR / "classifier"
    detector_root = config.DATA_DIR / "detector"
    directories = [classifier_root, detector_root]
    directories.extend(
        classifier_root / split / variety
        for split in ("train", "val", "test")
        for variety in config.VARIETY_CLASSES
    )
    directories.extend(
        detector_root / branch / split
        for branch in ("images", "labels")
        for split in ("train", "val", "test")
    )
    return directories


def _validate_split_dirs() -> None:
    """Validate all splitter destinations without following symlinks."""
    for directory in _split_dirs():
        _validate_real_dir(directory)


def _ensure_split_dirs() -> None:
    """Create splitter destinations only after validating every component."""
    directories = _split_dirs()
    _validate_split_dirs()
    for directory in directories:
        _ensure_real_dir(directory)


_SPLIT_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _validate_split_source_tree(root: Path, candidate_suffixes=()) -> list[Path]:
    """Reject source links and return regular candidate files at the root level."""
    root = _validate_real_dir(root)
    if not root.exists():
        return []

    candidates = []

    def walk(directory: Path, check_candidates: bool) -> None:
        for path in directory.iterdir():
            if path.is_symlink():
                raise HTTPException(status_code=400, detail=f"Split source must not be a symlink: {path}.")
            if check_candidates and path.suffix.lower() in candidate_suffixes and not path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Split source must be a regular file: {path}.",
                )
            if path.is_dir():
                walk(path, False)
            elif check_candidates and path.is_file() and path.suffix.lower() in candidate_suffixes:
                candidates.append(path)

    walk(root, True)
    return candidates


def _validate_split_sources() -> None:
    """Validate every raw source tree before the splitter can create outputs."""
    classifier_root = _classifier_raw()
    _validate_split_source_tree(classifier_root)
    if classifier_root.exists():
        for variety_dir in classifier_root.iterdir():
            if not variety_dir.is_symlink() and variety_dir.is_dir():
                _validate_split_source_tree(variety_dir, _SPLIT_IMAGE_EXTS)

    detector_images = _validate_split_source_tree(_detector_images(), _SPLIT_IMAGE_EXTS)
    _validate_split_source_tree(_detector_labels())
    for image in detector_images:
        label = _detector_labels() / f"{image.stem}.txt"
        if label.exists() and not label.is_file():
            raise HTTPException(status_code=400, detail=f"Split source must be a regular file: {label}.")


def _validate_split_file_targets() -> None:
    """Validate every file path that a split operation may overwrite."""
    classifier_root = config.DATA_DIR / "classifier"
    detector_root = config.DATA_DIR / "detector"
    splits = ("train", "val", "test")
    split_image_exts = _SPLIT_IMAGE_EXTS

    def validate_target(target: Path) -> None:
        if target.is_symlink():
            raise HTTPException(status_code=400, detail=f"Split destination must not be a symlink: {target}.")
        if target.exists() and not target.is_file():
            raise HTTPException(status_code=400, detail=f"Split destination must be a regular file: {target}.")

    for variety in config.VARIETY_CLASSES:
        source_root = _classifier_raw() / variety
        for source in _iter_regular_files(source_root, split_image_exts):
            for split in splits:
                validate_target(classifier_root / split / variety / source.name)

    for source in _iter_regular_files(_detector_images(), split_image_exts):
        label = _detector_labels() / f"{source.stem}.txt"
        for split in splits:
            validate_target(detector_root / "images" / split / source.name)
            if label.exists() or label.is_symlink():
                validate_target(detector_root / "labels" / split / label.name)


def _is_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _iter_regular_files(root: Path, suffixes=None):
    _validate_real_dir(root)
    if not root.exists():
        return
    for path in root.iterdir():
        if _is_regular_file(path) and (suffixes is None or path.suffix.lower() in suffixes):
            yield path


def _iter_regular_files_recursive(root: Path):
    _validate_real_dir(root)
    for path in root.iterdir():
        if _is_regular_file(path):
            yield path
        elif not path.is_symlink() and path.is_dir():
            yield from _iter_regular_files_recursive(path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_pixel_digest(image) -> str:
    """Hash normalized RGB pixels with their shape, independent of encoding."""
    shape_header = len(image.shape).to_bytes(1, "big") + b"".join(
        int(dimension).to_bytes(8, "big") for dimension in image.shape
    )
    return _sha256(shape_header + image.tobytes(order="C"))


def _existing_image_hashes() -> dict:
    hashes = {}
    roots = []
    for variety in config.VARIETY_CLASSES:
        roots.append(_classifier_raw() / variety)
    roots.append(_detector_images())
    for root in roots:
        for path in _iter_regular_files(root, IMAGE_EXTS):
            try:
                content = path.read_bytes()
                image = image_io.decode_image_bytes(content)
                hashes[_normalized_pixel_digest(image)] = str(path)
            except (OSError, ValueError):
                continue
    originals_dir = _originals_dir()
    if originals_dir.exists():
        for path in _iter_regular_files_recursive(originals_dir):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                content = path.read_bytes()
                image = image_io.decode_image_bytes(content)
                hashes[_normalized_pixel_digest(image)] = str(path)
            except (OSError, ValueError):
                continue
    return hashes


def _ear_numbers() -> set:
    numbers = set()
    roots = []
    for variety in config.VARIETY_CLASSES:
        roots.append(_classifier_raw() / variety)
    roots.append(_detector_images())
    for root in roots:
        for path in _iter_regular_files(root):
            match = re.match(r"^ear(\d+)$", parse_ear_id(path))
            if match:
                numbers.add(int(match.group(1)))
    originals_dir = _originals_dir()
    if originals_dir.exists():
        _validate_real_dir(originals_dir)
        for path in originals_dir.iterdir():
            if path.is_symlink():
                raise HTTPException(status_code=400, detail=f"Archive path must not be a symlink: {path}.")
            match = re.match(r"^ear(\d+)$", path.name)
            if match:
                numbers.add(int(match.group(1)))
    ground_truth_path = _gt_csv()
    if ground_truth_path.exists():
        for row in _read_ground_truth():
            match = re.match(r"^ear(\d+)$", row["ear_id"])
            if match:
                numbers.add(int(match.group(1)))
    return numbers


def _normalize_ear_id(raw: str) -> str:
    value = raw.strip()
    if not re.fullmatch(r"ear\d{3}", value):
        raise HTTPException(status_code=400, detail=f"Invalid ear id '{raw}'; expected 'earNNN'.")
    return value


def _next_ear_id() -> str:
    numbers = _ear_numbers()
    next_num = (max(numbers) + 1) if numbers else 1
    if next_num > 999:
        raise HTTPException(status_code=400, detail="No dataset ear IDs remain in the ear001–ear999 range.")
    return f"ear{next_num:03d}"


def _safe_original_name(filename: str) -> str:
    """Return only a safe basename while retaining the upload extension."""
    name = Path(str(filename or "").replace("\\", "/")).name.replace("\x00", "")
    return name if name not in {"", ".", ".."} else "upload"


def _archive_path(ear_id_value: str, filename: str, reserved_paths=()) -> Path:
    """Choose a non-colliding path in the exact archive directory for an ear."""
    archive_dir = _archive_dir(ear_id_value, create=False)
    if archive_dir is None:
        archive_dir = _originals_dir() / ear_id_value
    safe_name = _safe_original_name(filename)
    suffix = Path(safe_name).suffix
    stem = Path(safe_name).stem
    reserved_names = {Path(path).name.casefold() for path in reserved_paths}
    existing_names = {path.name.casefold() for path in archive_dir.iterdir()} if archive_dir.exists() else set()

    candidate = archive_dir / safe_name
    counter = 2
    while (
        candidate.is_symlink()
        or candidate.exists()
        or candidate.name.casefold() in existing_names
        or candidate.name.casefold() in reserved_names
    ):
        candidate = archive_dir / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def _stage_upload_file(target: Path, content: bytes, staged: list[tuple[Path, Path]]) -> None:
    """Write a request file to a same-directory temporary path."""
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    staged.append((temporary, target))
    temporary.write_bytes(content)


def _rollback_upload(staged: list[tuple[Path, Path]]) -> None:
    """Best-effort cleanup of temporary and committed files from one upload."""
    for temporary, target in staged:
        if _is_regular_file(temporary):
            try:
                temporary.unlink()
            except OSError:
                pass
        if _is_regular_file(target):
            try:
                target.unlink()
            except OSError:
                pass


def _commit_upload_batch(batch) -> None:
    """Commit several ears together, rolling the entire batch back on failure."""
    staged = []
    created_dirs = []
    try:
        originals_dir = _originals_dir()
        directories = _dataset_dirs()
        created_dirs = [directory for directory in directories if not directory.exists()]
        for entry in batch:
            archive_dir = originals_dir / entry["ear_id"]
            if not archive_dir.exists():
                created_dirs.append(archive_dir)

        _ensure_dirs()
        for entry in batch:
            _archive_dir(entry["ear_id"], create=True)

        for entry in batch:
            ear_id_value = entry["ear_id"]
            variety = entry["variety"]
            for plan in entry["plans"]:
                name = f"{ear_id_value}_v{plan['index']}.jpg"
                targets = [
                    (plan["archive_path"], plan["original"]),
                    (_classifier_raw() / variety / name, plan["canonical"]),
                    (_detector_images() / name, plan["canonical"]),
                ]
                if "label" in plan:
                    targets.append(
                        (_detector_labels() / f"{ear_id_value}_v{plan['index']}.txt", plan["label"])
                    )
                for target, content in targets:
                    _stage_upload_file(target, content, staged)

        for temporary, target in staged:
            temporary.replace(target)
    except Exception:
        _rollback_upload(staged)
        for directory in sorted(created_dirs, key=lambda path: len(path.parts), reverse=True):
            if directory.is_dir() and not directory.is_symlink():
                try:
                    directory.rmdir()
                except OSError:
                    pass
        raise


def _commit_upload(plans, variety: str, ear_id_value: str) -> None:
    """Commit one ear through the same atomic batch path used by bulk import."""
    _commit_upload_batch([{"ear_id": ear_id_value, "variety": variety, "plans": plans}])


def _commit_label_batch(plans) -> None:
    """Atomically create or replace a group of label files."""
    staged = []
    backups = []
    try:
        for target, content in plans:
            _stage_upload_file(target, content, staged)
        for temporary, target in staged:
            if target.exists():
                backup = target.with_name(f".{target.name}.bak-{uuid.uuid4().hex}")
                target.replace(backup)
                backups.append((backup, target))
            temporary.replace(target)
        for backup, _ in backups:
            try:
                backup.unlink()
            except OSError:
                pass
    except Exception:
        for temporary, target in staged:
            if _is_regular_file(temporary):
                try:
                    temporary.unlink()
                except OSError:
                    pass
            if _is_regular_file(target):
                try:
                    target.unlink()
                except OSError:
                    pass
        for backup, target in reversed(backups):
            if _is_regular_file(backup):
                try:
                    backup.replace(target)
                except OSError:
                    pass
        raise


def _validate_upload_targets(plans, variety: str, ear_id_value: str) -> None:
    for plan in plans:
        target_name = f"{ear_id_value}_v{plan['index']}.jpg"
        for root in (_classifier_raw() / variety, _detector_images()):
            existing = next(
                (
                    path
                    for path in _iter_regular_files(root)
                    if parse_ear_id(path) == ear_id_value
                    and parse_view_number(path) == plan["index"]
                    and path.suffix.lower() in IMAGE_EXTS
                ),
                None,
            )
            if existing is not None:
                raise HTTPException(
                    status_code=400, detail=f"{existing.name}: this view already exists for {ear_id_value}."
                )
            target = root / target_name
            if target.exists() or target.is_symlink():
                raise HTTPException(
                    status_code=400, detail=f"{target.name}: this view already exists for {ear_id_value}."
                )


def _grade(coverage, mold_count, insect_count):
    if mold_count > 0 or insect_count > 0 or coverage >= 0.10:
        return GRADE_REJECT
    if coverage >= 0.05:
        return GRADE_II
    if coverage > 0:
        return GRADE_I
    return GRADE_EXTRA


def _read_ground_truth():
    _validate_real_dir(_raw_dir())
    csv_path = _gt_csv()
    rows = []
    if csv_path.is_symlink():
        raise HTTPException(status_code=400, detail="Ground-truth file must not be a symlink.")
    if csv_path.exists():
        if not csv_path.is_file():
            raise HTTPException(status_code=400, detail="Ground-truth path must be a regular file.")
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 8:
                continue
            rows.append(
                {
                    "ear_id": parts[0],
                    "variety": parts[1],
                    "coverage": parts[2],
                    "mold_count": parts[3],
                    "insect_count": parts[4],
                    "missing_count": parts[5],
                    "other_count": parts[6],
                    "grade": parts[7],
                }
            )
    return rows


def _write_ground_truth(rows):
    _ensure_dirs()
    csv_path = _gt_csv()
    if csv_path.is_symlink():
        raise HTTPException(status_code=400, detail="Ground-truth file must not be a symlink.")
    if csv_path.exists() and not csv_path.is_file():
        raise HTTPException(status_code=400, detail="Ground-truth path must be a regular file.")
    header = "ear_id,variety,coverage,mold_count,insect_count,missing_count,other_count,grade\n"
    body = "".join(
        f"{r['ear_id']},{r['variety']},{r['coverage']},{r['mold_count']},"
        f"{r['insect_count']},{r['missing_count']},{r['other_count']},{r['grade']}\n"
        for r in rows
    )
    csv_path.write_text(header + body, encoding="utf-8")


@router.get("/stats")
def dataset_stats():
    _ensure_dirs()
    classifier = {}
    total_images = 0
    ears = {}

    for variety in config.VARIETY_CLASSES:
        vdir = _classifier_raw() / variety
        images = list(_iter_regular_files(vdir, IMAGE_EXTS))
        classifier[variety] = len(images)
        total_images += len(images)
        for p in images:
            key = parse_ear_id(p)
            entry = ears.setdefault(key, {"variety": variety, "views": set(), "labels": set()})
            entry["views"].add(parse_view_number(p))

    detector_images = sorted(
        _iter_regular_files(_detector_images(), IMAGE_EXTS)
    )
    boxes_per_class = [0] * len(config.DEFECT_CLASSES)
    missing_labels = []
    for img in detector_images:
        label = _detector_labels() / f"{img.stem}.txt"
        key = parse_ear_id(img)
        entry = ears.setdefault(key, {"variety": "", "views": set(), "labels": set()})
        entry["views"].add(parse_view_number(img))

        if _is_regular_file(label):
            entry["labels"].add(parse_view_number(img))
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    try:
                        cls_idx = int(parts[0])
                        if 0 <= cls_idx < len(config.DEFECT_CLASSES):
                            boxes_per_class[cls_idx] += 1
                    except ValueError:
                        continue
        else:
            missing_labels.append(img.name)

    incomplete = [
        {"ear_id": key, "views": len(info["views"])}
        for key, info in sorted(ears.items())
        if len(info["views"]) < config.VIEWS_PER_EAR
    ]

    return {
        "classifier_images": classifier,
        "classifier_total": total_images,
        "detector_images": len(detector_images),
        "detector_labels": len(detector_images) - len(missing_labels),
        "missing_labels": missing_labels,
        "boxes_per_class": dict(zip(config.DEFECT_CLASSES, boxes_per_class)),
        "ears": len(ears),
        "incomplete_ears": incomplete,
    }


@router.get("/ears")
def list_ears():
    _ensure_dirs()
    rows = []
    for variety in config.VARIETY_CLASSES:
        vdir = _classifier_raw() / variety
        for p in _iter_regular_files(vdir, IMAGE_EXTS):
            key = parse_ear_id(p)
            view = parse_view_number(p)
            existing = next((r for r in rows if r["ear_id"] == key), None)
            if existing is None:
                existing = {
                    "ear_id": key,
                    "variety": variety,
                    "views": [],
                    "labels": [],
                    "has_ground_truth": False,
                }
                rows.append(existing)
            existing["views"].append(view)
            if _is_regular_file(_detector_labels() / f"{key}_v{view}.txt"):
                existing["labels"].append(view)
    known = {r["ear_id"] for r in rows}
    known.update(_read_ground_truth_ear_ids())
    for row in rows:
        row["views"].sort()
        row["labels"].sort()
        row["has_ground_truth"] = row["ear_id"] in known
    return sorted(rows, key=lambda r: r["ear_id"])


def _read_ground_truth_ear_ids():
    return {row["ear_id"] for row in _read_ground_truth()}


@router.get("/validate")
def validate_dataset():
    _ensure_dirs()
    problems = []
    from training.validate_dataset import validate_classifier, validate_detector

    validate_classifier(_classifier_raw(), problems)
    validate_detector(_detector_images(), _detector_labels(), problems)
    return {"ok": not problems, "problems": problems}


@router.post("/ears")
async def upload_ear(
    ear_id: str = Form(...),
    variety: str = Form(...),
    images: List[UploadFile] = File(...),
    labels: List[UploadFile] = File(default=[]),
):
    _validate_dirs()
    if variety not in config.VARIETY_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown variety '{variety}'.")

    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if len(images) > config.VIEWS_PER_EAR:
        raise HTTPException(status_code=400, detail=f"At most {config.VIEWS_PER_EAR} images per ear.")
    if len(labels) > len(images):
        raise HTTPException(status_code=400, detail="More labels than images.")

    ear_id_value = _normalize_ear_id(ear_id) if ear_id != "auto" else _next_ear_id()

    existing_hashes = _existing_image_hashes()
    uploaded_hashes = set()
    archive_paths = set()

    plans = []
    for index, img in enumerate(images, start=1):
        content = await img.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Empty upload: {img.filename}")
        suffix = Path(img.filename or "").suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise HTTPException(status_code=400, detail=f"{img.filename}: unsupported type '{suffix}'.")
        try:
            normalized_image, canonical = image_io.normalize_image_bytes(content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{img.filename}: not a decodable image ({exc})")
        digest = _normalized_pixel_digest(normalized_image)
        if digest in uploaded_hashes:
            raise HTTPException(status_code=400, detail=f"{img.filename}: duplicated within this ear.")
        if digest in existing_hashes:
            raise HTTPException(
                status_code=400,
                detail=f"{img.filename}: already imported as {existing_hashes[digest]}.",
            )
        uploaded_hashes.add(digest)
        archive_path = _archive_path(ear_id_value, img.filename or "upload", archive_paths)
        archive_paths.add(archive_path)
        plans.append(
            {
                "index": index,
                "original": content,
                "canonical": canonical,
                "archive_path": archive_path,
            }
        )

    for index in range(1, len(images) + 1):
        target = _detector_labels() / f"{ear_id_value}_v{index}.txt"
        if target.exists() or target.is_symlink():
            raise HTTPException(
                status_code=400,
                detail=f"{target.name}: this view already has a label for {ear_id_value}.",
            )

    _validate_upload_targets(plans, variety, ear_id_value)
    problems = []
    for index, label in enumerate(labels, start=1):
        content = await label.read()
        text = content.decode("utf-8", errors="replace").strip()
        name = f"{ear_id_value}_v{index}.txt"
        check_label_lines(name, text.splitlines(), problems)
        plans[index - 1]["label"] = content
    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))

    _commit_upload(plans, variety, ear_id_value)

    return {"ear_id": ear_id_value, "views_added": len(plans), "labels_added": len(labels)}


@router.post("/bulk-ears")
async def upload_bulk_ears(
    variety: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """Import filename-grouped four-view ears as one atomic batch."""
    _validate_dirs()
    if variety not in config.VARIETY_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown variety '{variety}'.")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    grouped = {}
    for image in images:
        filename = Path(str(image.filename or "").replace("\\", "/")).name
        match = BULK_IMAGE_RE.fullmatch(filename)
        if match is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{filename or 'Unnamed file'}: expected the format "
                    "ear###_v1.heic through ear###_v4.png."
                ),
            )
        source_ear = match.group(1).lower()
        view = int(match.group(2))
        views = grouped.setdefault(source_ear, {})
        if view in views:
            raise HTTPException(status_code=400, detail=f"Duplicate view v{view} for {source_ear}.")
        views[view] = (image, filename)

    incomplete = []
    for source_ear, views in sorted(grouped.items()):
        missing = sorted(set(range(1, config.VIEWS_PER_EAR + 1)) - set(views))
        if missing:
            incomplete.append(f"{source_ear} missing " + ", ".join(f"v{view}" for view in missing))
    if incomplete:
        raise HTTPException(status_code=400, detail="Incomplete ear groups: " + "; ".join(incomplete))

    existing_hashes = _existing_image_hashes()
    uploaded_hashes = set()
    used_numbers = _ear_numbers()
    next_number = max(used_numbers, default=0) + 1
    batch = []

    for source_ear in sorted(grouped, key=lambda value: int(value[3:])):
        while next_number in used_numbers:
            next_number += 1
        if next_number > 999:
            raise HTTPException(status_code=400, detail="No dataset ear IDs remain in the ear001–ear999 range.")
        ear_id_value = f"ear{next_number:03d}"
        used_numbers.add(next_number)
        next_number += 1
        archive_paths = set()
        plans = []
        for view in range(1, config.VIEWS_PER_EAR + 1):
            upload, filename = grouped[source_ear][view]
            content = await upload.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"Empty upload: {filename}")
            try:
                normalized_image, canonical = image_io.normalize_image_bytes(content)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{filename}: not a decodable image ({exc})")
            digest = _normalized_pixel_digest(normalized_image)
            if digest in uploaded_hashes:
                raise HTTPException(status_code=400, detail=f"{filename}: duplicated in this batch.")
            if digest in existing_hashes:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename}: already imported as {existing_hashes[digest]}.",
                )
            uploaded_hashes.add(digest)
            archive_path = _archive_path(ear_id_value, filename, archive_paths)
            archive_paths.add(archive_path)
            plans.append(
                {
                    "index": view,
                    "original": content,
                    "canonical": canonical,
                    "archive_path": archive_path,
                }
            )
        _validate_upload_targets(plans, variety, ear_id_value)
        batch.append({"ear_id": ear_id_value, "variety": variety, "plans": plans})

    _commit_upload_batch(batch)
    return {
        "ears_imported": len(batch),
        "views_imported": len(images),
        "ear_ids": [entry["ear_id"] for entry in batch],
    }


@router.post("/ears/{ear_id}/labels")
async def attach_ear_label(ear_id: str, view: int = Form(...), label: UploadFile = File(...)):
    _ensure_dirs()
    ear_id_value = _normalize_ear_id(ear_id)
    if not 1 <= view <= config.VIEWS_PER_EAR:
        raise HTTPException(status_code=400, detail=f"View must be between 1 and {config.VIEWS_PER_EAR}.")
    if not any(
        parse_ear_id(p) == ear_id_value and parse_view_number(p) == view and p.suffix.lower() in IMAGE_EXTS
        for p in _iter_regular_files(_detector_images(), IMAGE_EXTS)
    ):
        raise HTTPException(status_code=404, detail=f"No image imported for {ear_id_value} view {view}.")
    target = _detector_labels() / f"{ear_id_value}_v{view}.txt"
    if target.exists() or target.is_symlink():
        raise HTTPException(status_code=400, detail=f"Label already exists: {target.name}.")
    content = await label.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty label file.")
    text = content.decode("utf-8", errors="replace").strip()
    problems = []
    check_label_lines(target.name, text.splitlines(), problems)
    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))
    _commit_label_batch([(target, content)])
    return {"ear_id": ear_id_value, "view": view, "written": target.name}


@router.put("/ears/{ear_id}/labels/bulk")
async def replace_ear_labels_bulk(ear_id: str, request: Request):
    return await _handle_bulk_labels(ear_id, request, overwrite=True)


@router.put("/ears/{ear_id}/labels/{view}")
async def replace_ear_label(ear_id: str, view: int, label: UploadFile = File(...)):
    _ensure_dirs()
    ear_id_value = _normalize_ear_id(ear_id)
    if not 1 <= view <= config.VIEWS_PER_EAR:
        raise HTTPException(status_code=400, detail=f"View must be between 1 and {config.VIEWS_PER_EAR}.")
    _view_image_path(ear_id_value, view)

    target = _detector_labels() / f"{ear_id_value}_v{view}.txt"
    if target.is_symlink():
        raise HTTPException(status_code=400, detail=f"Label path must not be a symlink: {target.name}.")
    if target.exists() and not target.is_file():
        raise HTTPException(status_code=400, detail=f"Label path must be a regular file: {target.name}.")

    content = await label.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty label file.")
    text = content.decode("utf-8", errors="replace").strip()
    problems = []
    check_label_lines(target.name, text.splitlines(), problems)
    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))

    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(content)
        temporary.replace(target)
    except Exception:
        if _is_regular_file(temporary):
            try:
                temporary.unlink()
            except OSError:
                pass
        raise
    return {"ear_id": ear_id_value, "view": view, "written": target.name}


async def _handle_bulk_labels(ear_id: str, request: Request, overwrite: bool):
    _ensure_dirs()
    ear_id_value = _normalize_ear_id(ear_id)
    form = await request.form()
    views_raw = form.getlist("views")
    labels = form.getlist("labels")
    if len(views_raw) != len(labels):
        raise HTTPException(status_code=400, detail="Mismatch between views and label files.")
    if not views_raw:
        raise HTTPException(status_code=400, detail="No labels to attach.")

    views = []
    for raw in views_raw:
        try:
            views.append(int(raw))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"View must be an integer, got '{raw}'.")
    if len(set(views)) != len(views):
        raise HTTPException(status_code=400, detail="Duplicate view numbers are not allowed.")
    for view in views:
        if not 1 <= view <= config.VIEWS_PER_EAR:
            raise HTTPException(
                status_code=400, detail=f"View must be between 1 and {config.VIEWS_PER_EAR}."
            )

    image_views = {
        parse_view_number(p)
        for p in _iter_regular_files(_detector_images(), IMAGE_EXTS)
        if parse_ear_id(p) == ear_id_value
    }

    problems = []
    plans = []
    for view, label in zip(views, labels):
        if view not in image_views:
            problems.append(f"No image imported for {ear_id_value} view {view}.")
            continue
        target = _detector_labels() / f"{ear_id_value}_v{view}.txt"
        if target.is_symlink():
            problems.append(f"Label path must not be a symlink: {target.name}.")
            continue
        if target.exists() and not target.is_file():
            problems.append(f"Label path must be a regular file: {target.name}.")
            continue
        if target.exists() and not overwrite:
            problems.append(f"Label already exists: {target.name}.")
            continue
        content = await label.read()
        if not content:
            problems.append(f"Empty label file for view {view}.")
            continue
        text = content.decode("utf-8", errors="replace").strip()
        check_label_lines(target.name, text.splitlines(), problems)
        plans.append((target, content))

    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))

    _commit_label_batch(plans)
    written = [target.name for target, _ in plans]
    return {"ear_id": ear_id_value, "written": written, "attached": len(written)}


@router.post("/ears/{ear_id}/labels/bulk")
async def attach_ear_labels_bulk(ear_id: str, request: Request):
    return await _handle_bulk_labels(ear_id, request, overwrite=False)


def _view_image_path(ear_id_value: str, view: int) -> Path:
    for p in _iter_regular_files(_detector_images(), IMAGE_EXTS):
        if parse_ear_id(p) == ear_id_value and parse_view_number(p) == view:
            return p
    raise HTTPException(status_code=404, detail=f"No image for {ear_id_value} view {view}.")


def _media_type(path: Path) -> str:
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "heic": "image/heic",
        "heif": "image/heif",
    }.get(
        path.suffix.lower().lstrip("."), "application/octet-stream"
    )


@router.get("/ears/{ear_id}/views/{view}/image")
def get_view_image(ear_id: str, view: int):
    if not 1 <= view <= config.VIEWS_PER_EAR:
        raise HTTPException(status_code=400, detail=f"View must be between 1 and {config.VIEWS_PER_EAR}.")
    path = _view_image_path(_normalize_ear_id(ear_id), view)
    return FileResponse(str(path), media_type=_media_type(path))


@router.get("/ears/{ear_id}/views/{view}/label")
def get_view_label(ear_id: str, view: int):
    if not 1 <= view <= config.VIEWS_PER_EAR:
        raise HTTPException(status_code=400, detail=f"View must be between 1 and {config.VIEWS_PER_EAR}.")
    _validate_real_dir(_detector_labels())
    target = _detector_labels() / f"{_normalize_ear_id(ear_id)}_v{view}.txt"
    if not _is_regular_file(target):
        raise HTTPException(status_code=404, detail=f"No annotation for {_normalize_ear_id(ear_id)} view {view}.")
    return FileResponse(str(target), media_type="text/plain")


@router.delete("/ears/{ear_id}")
def delete_ear(ear_id: str):
    _ensure_dirs()
    ear_id_value = _normalize_ear_id(ear_id)
    archive_dir = _archive_dir(ear_id_value, create=False)

    label_targets = []
    for view in range(1, config.VIEWS_PER_EAR + 1):
        target = _detector_labels() / f"{ear_id_value}_v{view}.txt"
        if _is_regular_file(target):
            label_targets.append(target)

    removed_images = 0
    removed_labels = 0
    removed_originals = 0

    canonical_names = {
        f"{ear_id_value}_v{view}.jpg" for view in range(1, config.VIEWS_PER_EAR + 1)
    }
    for variety in config.VARIETY_CLASSES:
        vdir = _classifier_raw() / variety
        for p in list(_iter_regular_files(vdir)):
            if p.name in canonical_names:
                p.unlink()
                removed_images += 1
    for p in list(_iter_regular_files(_detector_images())):
        if p.name in canonical_names:
            p.unlink()
            removed_images += 1
    for target in label_targets:
        target.unlink()
        removed_labels += 1

    if archive_dir is not None and not archive_dir.is_symlink() and archive_dir.is_dir():
        removed_originals = sum(1 for _ in _iter_regular_files_recursive(archive_dir))
        shutil.rmtree(archive_dir)

    rows = [r for r in _read_ground_truth() if r["ear_id"] != ear_id_value]
    removed_gt = len(_read_ground_truth()) - len(rows)
    _write_ground_truth(rows)

    return {
        "ear_id": ear_id_value,
        "removed_images": removed_images,
        "removed_labels": removed_labels,
        "removed_originals": removed_originals,
        "removed_ground_truth": removed_gt,
    }


@router.get("/ground-truth")
def get_ground_truth():
    return {"rows": _read_ground_truth()}


@router.put("/ground-truth")
def upsert_ground_truth(payload: dict):
    _ensure_dirs()
    ear_id_value = _normalize_ear_id(payload.get("ear_id", ""))
    variety = payload.get("variety")
    if variety not in config.VARIETY_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown variety '{variety}'.")
    try:
        coverage = float(payload["coverage"])
        counts = {
            "mold_count": int(payload["mold_count"]),
            "insect_count": int(payload["insect_count"]),
            "missing_count": int(payload["missing_count"]),
            "other_count": int(payload["other_count"]),
        }
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Coverage and count fields must be numbers.")
    if not 0.0 <= coverage <= 1.0:
        raise HTTPException(status_code=400, detail="Coverage must be in [0, 1].")
    for name, value in counts.items():
        if value < 0:
            raise HTTPException(status_code=400, detail=f"{name} must be >= 0.")

    row = {
        "ear_id": ear_id_value,
        "variety": variety,
        "coverage": f"{coverage:.4g}",
        "mold_count": counts["mold_count"],
        "insect_count": counts["insect_count"],
        "missing_count": counts["missing_count"],
        "other_count": counts["other_count"],
        "grade": _grade(coverage, counts["mold_count"], counts["insect_count"]),
    }
    rows = [r for r in _read_ground_truth() if r["ear_id"] != ear_id_value]
    rows.append(row)
    rows.sort(key=lambda r: r["ear_id"])
    _write_ground_truth(rows)
    return {"row": row}


@router.post("/split")
def make_split(confirm: bool = False):
    if not confirm:
        _ensure_dirs()
        raise HTTPException(
            status_code=400, detail="Pass ?confirm=true to overwrite existing split folders."
        )
    # Preflight raw and split paths together so a rejected output path cannot
    # leave newly created dataset directories behind.
    _validate_dirs()
    _validate_split_sources()
    _validate_split_dirs()
    _validate_split_file_targets()
    _ensure_dirs()
    _ensure_split_dirs()
    cls_counts = split_classifier(_classifier_raw(), config.DATA_DIR, seed=42)
    det_counts = split_detector(_detector_images(), _detector_labels(), config.DATA_DIR / "detector", 42)
    return {"classifier": cls_counts, "detector": det_counts}
