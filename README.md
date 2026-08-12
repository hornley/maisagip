# Maisagip

A Hybrid Deep Learning Framework for Corn Ear Quality Grading and Defect Detection Using Computer Vision (CS 307 Thesis 1).

Full pipeline skeleton: FastAPI backend, EfficientNetV2-S variety classifier, YOLOv11n defect detector, multi-view ear merge, PNS/BAFS-based coverage rule engine, mobile-responsive web frontend, training harness, and evaluation scripts. The web app runs end-to-end today in **demo mode** (heuristic providers) and switches to the trained models automatically once weights are dropped in.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Open http://localhost:8000, upload one photo — or all 4 roll views of an ear — and inspect. `GET /health` shows which providers are active.

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
| Below Class II (Reject) | ≥ 10% |

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

**Detector class order (fixed — matches `config.DEFECT_CLASSES`):**

```
0  corn_ear      3  discoloration
1  mold          4  deformity
2  insect_damage 5  missing_kernels
```

Annotation protocol:
- One `corn_ear` box per view (drives size/completeness).
- Box every visible defect, in **every view where it appears**, same class per physical spot.
- Whole-ear defects (deformity, discoloration) get ear-sized boxes — coverage % handles them.
- YOLO export: `class x y w h` (normalized), filename = image stem.

```bash
data/raw/classifier/{white_corn,yellow_sweet_corn}/ear001_v1.jpg ...
data/raw/detector/images/ear001_v1.jpg ...
data/raw/detector/labels/ear001_v1.txt ...
```

Validate → split → train:

```bash
python -m training.validate_dataset
python -m training.make_dataset_split        # leak-free per-ear 80/10/10 splits
pip install -r requirements-ml.txt
python -m training.train_classifier --epochs 30
python -m training.train_detector --epochs 100
```

Weights land in `data/weights/` and the app auto-switches to real mode.

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