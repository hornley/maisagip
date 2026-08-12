import hashlib
import io
import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from . import config
from training.validate_dataset import check_label_lines, ear_id, view_number
from training.make_dataset_split import split_classifier, split_detector

router = APIRouter(prefix="/dataset", tags=["dataset"])

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

GRADE_EXTRA = "Extra Class"
GRADE_I = "Class I"
GRADE_II = "Class II"
GRADE_REJECT = "Reject"


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


def _gt_csv() -> Path:
    return _raw_dir() / "ground_truth.csv"


def _ensure_dirs():
    for variety in config.VARIETY_CLASSES:
        (_classifier_raw() / variety).mkdir(parents=True, exist_ok=True)
    _detector_images().mkdir(parents=True, exist_ok=True)
    _detector_labels().mkdir(parents=True, exist_ok=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _existing_image_hashes() -> dict:
    hashes = {}
    roots = []
    if _classifier_raw().exists():
        for variety in config.VARIETY_CLASSES:
            roots.append(_classifier_raw() / variety)
    if _detector_images().exists():
        roots.append(_detector_images())
    for root in roots:
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                try:
                    hashes[_sha256(path.read_bytes())] = str(path)
                except OSError:
                    continue
    return hashes


def _ear_numbers() -> set:
    numbers = set()
    roots = []
    if _classifier_raw().exists():
        for variety in config.VARIETY_CLASSES:
            roots.append(_classifier_raw() / variety)
    if _detector_images().exists():
        roots.append(_detector_images())
    for root in roots:
        for path in root.iterdir():
            match = re.match(r"^ear(\d+)$", ear_id(path))
            if match:
                numbers.add(int(match.group(1)))
    return numbers


def _normalize_ear_id(raw: str) -> str:
    value = raw.strip()
    if not re.match(r"^ear\d+$", value):
        raise HTTPException(status_code=400, detail=f"Invalid ear id '{raw}'; expected 'earNNN'.")
    return value


def _next_ear_id() -> str:
    numbers = _ear_numbers()
    next_num = (max(numbers) + 1) if numbers else 1
    return f"ear{next_num:03d}"


def _decode_check(data: bytes, name: str):
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{name}: not a decodable image ({exc})")


def _grade(coverage, mold_count, insect_count):
    if mold_count > 0 or insect_count > 0 or coverage >= 0.10:
        return GRADE_REJECT
    if coverage >= 0.05:
        return GRADE_II
    if coverage > 0:
        return GRADE_I
    return GRADE_EXTRA


def _read_ground_truth():
    csv_path = _gt_csv()
    rows = []
    if csv_path.exists():
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
    _raw_dir().mkdir(parents=True, exist_ok=True)
    header = "ear_id,variety,coverage,mold_count,insect_count,missing_count,other_count,grade\n"
    body = "".join(
        f"{r['ear_id']},{r['variety']},{r['coverage']},{r['mold_count']},"
        f"{r['insect_count']},{r['missing_count']},{r['other_count']},{r['grade']}\n"
        for r in rows
    )
    _gt_csv().write_text(header + body, encoding="utf-8")


@router.get("/stats")
def dataset_stats():
    _ensure_dirs()
    classifier = {}
    total_images = 0
    ears = {}

    for variety in config.VARIETY_CLASSES:
        vdir = _classifier_raw() / variety
        images = [p for p in vdir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        classifier[variety] = len(images)
        total_images += len(images)
        for p in images:
            key = ear_id(p)
            entry = ears.setdefault(key, {"variety": variety, "views": set(), "labels": set()})
            entry["views"].add(view_number(p))

    detector_images = sorted(
        p for p in _detector_images().iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    boxes_per_class = [0] * len(config.DEFECT_CLASSES)
    missing_labels = []
    for img in detector_images:
        label = _detector_labels() / f"{img.stem}.txt"
        key = ear_id(img)
        entry = ears.setdefault(key, {"variety": "", "views": set(), "labels": set()})
        entry["views"].add(view_number(img))
        if label.exists():
            entry["labels"].add(view_number(img))
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
        for p in vdir.iterdir():
            if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
                continue
            key = ear_id(p)
            view = view_number(p)
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
            if (_detector_labels() / f"{key}_v{view}.txt").exists():
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
    _ensure_dirs()
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

    plans = []
    for index, img in enumerate(images, start=1):
        content = await img.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Empty upload: {img.filename}")
        suffix = Path(img.filename or "").suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise HTTPException(status_code=400, detail=f"{img.filename}: unsupported type '{suffix}'.")
        _decode_check(content, img.filename)
        digest = _sha256(content)
        if digest in uploaded_hashes:
            raise HTTPException(status_code=400, detail=f"{img.filename}: duplicated within this ear.")
        if digest in existing_hashes:
            raise HTTPException(
                status_code=400,
                detail=f"{img.filename}: already imported as {existing_hashes[digest]}.",
            )
        uploaded_hashes.add(digest)
        plans.append({"index": index, "suffix": suffix, "content": content})

    for index in range(1, len(images) + 1):
        target = _detector_labels() / f"{ear_id_value}_v{index}.txt"
        if target.exists():
            raise HTTPException(
                status_code=400,
                detail=f"{target.name}: this view already has a label for {ear_id_value}.",
            )

    problems = []
    for index, img in enumerate(images, start=1):
        target_img = _detector_images() / f"{ear_id_value}_v{index}{plans[index - 1]['suffix']}"
        if target_img.exists():
            raise HTTPException(
                status_code=400, detail=f"{target_img.name}: this view already exists for {ear_id_value}."
            )
    for index, label in enumerate(labels, start=1):
        content = await label.read()
        text = content.decode("utf-8", errors="replace").strip()
        name = f"{ear_id_value}_v{index}.txt"
        check_label_lines(name, text.splitlines(), problems)
    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))

    for plan in plans:
        name = f"{ear_id_value}_v{plan['index']}{plan['suffix']}"
        (_classifier_raw() / variety / name).write_bytes(plan["content"])
        (_detector_images() / name).write_bytes(plan["content"])
    for index, label in enumerate(labels, start=1):
        content = await label.read()
        (_detector_labels() / f"{ear_id_value}_v{index}.txt").write_bytes(content)

    return {"ear_id": ear_id_value, "views_added": len(plans), "labels_added": len(labels)}


@router.get("/ground-truth")
def get_ground_truth():
    return {"rows": _read_ground_truth()}


@router.put("/ground-truth")
def upsert_ground_truth(payload: dict):
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
        raise HTTPException(
            status_code=400, detail="Pass ?confirm=true to overwrite existing split folders."
        )
    cls_counts = split_classifier(_classifier_raw(), config.DATA_DIR, seed=42)
    det_counts = split_detector(_detector_images(), _detector_labels(), config.DATA_DIR / "detector", 42)
    return {"classifier": cls_counts, "detector": det_counts}
