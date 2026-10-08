# Maisagip

A Hybrid Deep Learning Framework for Corn Ear Quality Grading and Defect Detection Using Computer Vision (CS 307 Thesis 1).

Full pipeline skeleton: FastAPI backend, EfficientNetV2-S variety classifier, YOLOv11n defect detector, multi-view ear merge, PNS/BAFS-based coverage rule engine, mobile-responsive web frontend, training harness, and evaluation scripts. The web app runs end-to-end today in **demo mode** (heuristic providers) and switches to the trained models automatically once weights are dropped in.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Open http://localhost:8000, upload one photo — or all 4 roll views of an ear — and inspect. HEIC, HEIF, JPG, JPEG, and PNG inputs are accepted. `GET /health` shows which providers are active.

## Architecture

```
backend/app/
├── main.py       POST /inspect (multi-file), GET /report/<id>(/views/<n>/image), /health, / (frontend)
├── pipeline.py   decode each view -> classify -> detect -> traits -> merge -> rules -> report
├── models.py     EfficientNetV2-S + YOLOv11n loaders with demo fallback
├── views.py      multi-view merge: variety vote, defect dedup, coverage %, traits
├── traits.py     ear size + kernel completeness from detection geometry
├── rules/        coverage-based RDSS engine + config_rules.json (PNS-based)
└── report.py     report JSON + per-view annotated images + grid
```

Pipeline stages are independent, testable units. Heavy ML stacks (`torch`, `ultralytics`) are lazy-imported only when real weights exist, so the web app and tests run on the light requirements alone.

## Inspection flow

1. Upload 1..N photos of an ear (recommended: 4 roll views, 0°/90°/180°/270°).
2. Each view is independently classified for variety and scanned for defects.
3. `views.py` merges the views into one ear-level result:
   - **Variety:** majority vote; ties break on confidence.
   - **Defects:** the same physical defect seen in adjacent views is deduplicated by size similarity → merged count; each class reports coverage, confidence, and which views show it.
   - **Defect coverage %:** Σ per-physical-defect largest-area / largest ear-box area.
   - Kernel completeness = worst (min) across views; ear size from the longest ear box.
4. The rule engine assigns grade + utilization recommendation, the report saves per-view annotated images + a grid, and the UI shows provenance ("missing_kernels · views 3,4").

## Grading rules (PNS-based, in `rules/config_rules.json`)

| Grade | Defect coverage |
| --- | --- |
| Extra Class | 0% |
| Class I | < 5% |
| Class II | < 10% |
| Reject | ≥ 10% |

Utilization overrides: coverage ≥ 10% → reject; any mold/insect damage → animal feed (non-human); else < 5% → human consumption; < 10% → food processing. Numbers in the JSON are your authoritative thresholds verbatim, so calibrating later requires zero code changes.

## Demo mode vs. real models

| Provider | When active |
| --- | --- |
| Demo classifier (color heuristic) | `data/weights/efficientnetv2_s_corn.pt` missing or fails to load |
| Real EfficientNetV2-S | weights present |
| Demo detector (rule-based boxes) | `data/weights/corn_yolov11n.pt` missing |
| Real YOLOv11n | weights present |

## Dataset conventions

Capture protocol (per ear): fixed camera + turntable; roll the ear 90° between shots → `ear001_v1.jpg … ear001_v4.jpg`. The resting contact band on the display is never visible — accepted as a documented delimitation.

HEIC/HEIF inputs are decoded with `pillow-heif`, orientation-corrected, converted to RGB, and stored for training as canonical JPEGs. The untouched upload is archived under `data/raw/originals/` with its dataset-relative path. Install this support with the normal dependencies:

```bash
pip install -r requirements.txt
```

To migrate HEIC/HEIF files already in the raw dataset, run:

```bash
python -m training.normalize_heic_dataset --input-root data/raw --originals-root data/raw/originals
```

The migration never removes HEIC/HEIF files or overwrites an existing JPEG. It requires POSIX descriptor-relative no-follow filesystem I/O and fails closed on platforms without it. Corrupt files are reported and must be reviewed before training.

**Detector class order (fixed — matches `config.DEFECT_CLASSES`):**

```
0  corn_ear      3  discoloration
1  mold          4  deformity
2  insect_damage 5  missing_kernels
                       6  ear_decay
                       7  shriveled_kernels
```

`shriveled_kernels` is appended to preserve the meaning of existing class IDs
and detector outputs. Existing seven-class detector weights remain compatible
with their original classes, but the detector must be retrained with the
expanded dataset to predict the new class.

Annotation protocol:
- One `corn_ear` box per view (drives size/completeness).
- Box every visible defect, in **every view where it appears**, same class per physical spot.
- Deformity and extensive discoloration get ear-sized boxes — coverage % handles them.
- Box visibly dried or shriveled kernel regions as `shriveled_kernels`, using a tight box around the affected patch rather than the whole ear unless the defect truly spans the whole ear.
- YOLO export: `class x y w h` (normalized), filename = image stem.

The Dataset page includes a local annotation desk. Click **Draw boxes** for an imported ear, choose a class, drag boxes over the processed image, and save each view. Existing YOLO labels can be edited in place; the tool requires a `corn_ear` box before saving. To replace an uploaded `.txt`, expand **Annotate**, choose **replace .txt**, and click **Save labels**.

```bash
data/raw/classifier/{white_corn,yellow_sweet_corn}/ear001_v1.jpg ...
data/raw/detector/images/ear001_v1.jpg ...
data/raw/detector/labels/ear001_v1.txt ...
```

Validate → split → train:

```bash
python -m training.validate_dataset
python -m training.make_dataset_split --seed 42  # leak-free ear-level 80/10/10 split
pip install -r requirements-ml.txt
python -m training.train_classifier --epochs 30
python -m training.train_detector --epochs 100
```

The existing single-model detector command uses `yolo11n.pt` by default, writes its
dataset definition to `data/detector/data.yaml`, and copies only its best checkpoint
to `data/weights/corn_yolov11n.pt`. That is the production training path; comparison
runs do not change or populate `data/weights/`. The app auto-switches to real mode
when production weights are present.

### YOLO comparison experiment

The comparison catalog is exactly:

| Family | Models | Training implementation |
| --- | --- | --- |
| YOLOv7 | `yolov7-tiny`, `yolov7`, `yolov7x` | Official YOLOv7 checkout, using its native `train.py` and `test.py` |
| YOLOv8/11 | `yolov8n`, `yolov8s`, `yolov8m`, `yolo11s` | Ultralytics |

Prepare the shared detector split and run all seven models with:

```bash
python -m training.validate_dataset
python -m training.make_dataset_split --seed 42
python -m training.yolo_comparison \
  --epochs 100 --batch 16 --imgsz 640 --seed 42 \
  --yolov7-root /path/to/yolov7
```

The split is made at ear level: all four views of an ear stay together in train,
validation, or test, so the models use the same leak-free split. Results and
comparison checkpoints are isolated under `data/comparisons/yolo/`. The runner may
regenerate the shared `data/detector/data.yaml`, but it does not modify the source
detector images or labels and never copies comparison checkpoints into production
`data/weights/`.

Use `--dry-run` to inspect every planned command without importing the heavy ML
packages or starting training:

```bash
python -m training.yolo_comparison \
  --seed 42 --yolov7-root /path/to/yolov7 --dry-run
```

To run only one model, use the comma-separated `--models` filter. Ultralytics-only
runs do not need a YOLOv7 checkout:

```bash
python -m training.yolo_comparison \
  --models yolo11s --epochs 100 --batch 16 --imgsz 640 --seed 42
```

For YOLOv7, `--yolov7-python /path/to/yolov7-venv/bin/python` selects the Python
environment for the native scripts, and `--yolov7-weights-dir /path/to/weights`
selects local files such as `yolov7x.pt` instead of relying on the native download.
The official YOLOv7 `train.py` does not accept `--seed`. The runner records the
requested seed and `PYTHONHASHSEED` in run metadata, warns that YOLOv7's internal
seed remains hardcoded, and treats those results as exploratory rather than claiming
identical internal seeding across all models.

The current dataset contains only 11 ears / 44 views. These runs are therefore
exploratory comparisons for pipeline development, not reliable or publishable
benchmarks; conclusions should be revisited after collecting substantially more
ears.

## Ground truth & eval

Per-ear ground truth CSV at `data/raw/ground_truth.csv`:

```
ear_id,variety,coverage,mold_count,insect_count,missing_count,other_count,grade
ear001,yellow_sweet_corn,0.03,1,0,0,0,Class I
```

```bash
python -m eval.eval_merge            # model coverage/grade vs. human CSV
python -m eval.eval_classifier       # classifier accuracy/precision/recall/F1
python -m eval.eval_detector         # mAP@0.5, mAP@0.5:0.95, IoU, latency
```

## Tests

```bash
python -m pytest backend/tests -v
```

Covers the merge truth table (dedup across views, coverage, whole-ear defects, tie-breaks), coverage grades, per-ear leak-free splitting, dataset validation, and the multi-view API end-to-end.

## Scope

RGB images only; externally visible traits only (no moisture, sweetness, aflatoxin, etc.). Inspection assists post-harvest grading decisions and does not replace official PNS grading by authorized agencies.
