# Maisagip — Full Pipeline Skeleton — Design

Date: 2026-08-07
Project: Maisagip — A Hybrid Deep Learning Framework for Corn Ear Quality Grading and Defect Detection Using Computer Vision (CS 307 Thesis 1).

## Goal

Build a working end-to-end skeleton of the Maisagip system so the web app, pipeline, rule engine, reports, training harness, and evaluation scripts all run today — with real model weights swappable in later when the annotated dataset is ready.

## Architecture

Modular monorepo. All stages are independent, testable units:

```
maisagip/
├── backend/            # FastAPI app + pipeline orchestration
│   ├── app/
│   │   ├── main.py             # routes: POST /inspect, GET /, /health, /report/<id>, /report/<id>/image
│   │   ├── pipeline.py         # orchestrates: decode → classify → detect → traits → rules → report
│   │   ├── config.py           # paths, model weight locations, class lists, thresholds
│   │   ├── models.py           # model loaders with automatic "demo provider" fallback
│   │   ├── traits.py           # ear size + kernel completeness derived from detector geometry
│   │   ├── rules/              # RDSS engine + config_rules.json (PNS/BAFS-based)
│   │   └── report.py           # builds inspection report (JSON + annotated image)
│   └── tests/
├── frontend/           # static mobile-responsive UI (HTML/CSS/JS, no build step)
├── training/           # make_dataset_split.py, train_classifier.py, train_detector.py
├── eval/               # eval_classifier.py, eval_detector.py
├── data/               # drop-in dataset folders (see Dataset conventions)
├── requirements.txt    # runtime deps (light)
├── requirements-ml.txt # training/eval deps (torch, ultralytics)
└── README.md
```

## Components

### 1. Model layer (`models.py`)
- **Classifier**: `EfficientNetV2-S` (torchvision, 2 output classes: `white_corn`, `yellow_sweet_corn`). Weights expected at `data/weights/efficientnetv2_s_corn.pt`.
- **Detector**: `Ultralytics YOLOv11n` with 6 classes: `corn_ear`, `mold`, `insect_damage`, `discoloration`, `deformity`, `missing_kernels`. Weights expected at `data/weights/corn_yolov11n.pt`.
- **Demo fallback**: if weights are absent OR torch/ultralytics are not importable, a deterministic heuristic provider returns plausible outputs so the whole web flow is demonstrable immediately. A `mode` field on the report records which provider ran.

### 2. Pipeline data flow (`pipeline.py`)
Upload → decode/validate → RGB normalize → YOLO detects defects + corn ear (boxes + confidence) → EfficientNetV2 classifies variety → `traits.py` derives ear size (ear-box length vs. config buckets) and kernel completeness (1 − missing-kernel defect area / ear area) → RD engine applies PNS/BAFS rules → report (JSON + annotated image saved under `data/reports/`).

### 3. Rule engine (`rules/`) — PNS/BAFS 98:2022 basis
Rules are data, in `config_rules.json`:
- Grades `Extra Class`, `Class I`, `Class II`, each with `max_severe`, `max_minor`, `min_completeness`.
- Ordered utilization rules (`always` / `any_severe` / `any_minor` / `completeness_lt`) mapping to `human_consumption`, `food_processing`, `animal_feed`, `reject`.
- Engine outputs the assigned grade (or "Below Class II") plus the **exact rules that fired** (explainability) and a `needs_reinspection` flag when confidences are low.

Threshold numbers are an initial encoding; a PNS calibration section in README documents how to align them to the official standard tables.

### 4. Report (`report.py`)
Inspection report contains: report id, inspection date, variety + confidence, detected defects (class, box, confidence), estimated ear size + kernel completeness, assigned grade, utilization recommendation, overall confidence, provider mode, and a saved annotated image. A module-level in-memory store maps id → report for the demo; the image is persisted to `data/reports/<id>.png`.

### 5. API (`main.py`)
- `POST /inspect` — multipart image upload → full pipeline → report JSON (anchored annotated image referenced under `/report/<id>/image`).
- `GET /` — serves the responsive frontend.
- `GET /report/<id>` — report metadata; `GET /report/<id>/image` — annotated PNG.
- `GET /health` — provider mode, model readiness.

### 6. Frontend (`frontend/`)
Standalone static assets served by FastAPI. Mobile-first, no build step: upload → progress → inspection card (variety, confidence bars, defect chips, ear size, completeness, big grade, recommendation, reason chips, date). Annotated image displayed with uploaded preview.

### 7. Training (`training/`)
- `make_dataset_split.py` — copies raw photos into standard train/val/test trees for classifier and detector.
- `train_classifier.py` — torchvision EfficientNetV2-S fine-tune (device auto).
- `train_detector.py` — Ultralytics YOLOv11n single-class training (device auto).
- Model weights saved under `data/weights/`.

### 8. Eval (`eval/`)
- `eval_classifier.py` — accuracy, precision, recall, F1 (macro + per class).
- `eval_detector.py` — mAP@0.5, mAP@0.5:0.95, IoU, inference time.

### 9. Testing
- `test_rules.py` — truth-table over Engine with synthetic defect inputs.
- `test_traits.py` — ear size bucketing + completeness math.
- `test_pipeline.py` — end-to-end inspect via TestClient with demo provider on a generated image; asserts required report fields.

## Dataset conventions

- Classifier drop-in: `data/alternates/classifier/{train,val,test}/white_corn|yellow_sweet_corn/*.{jpg,png}`
- Detector drop-in: `data/alternatives/detector/images/{train,val,test}/*.jpg` + `data/alternatives/detector/labels/{train,val,test}/*.txt` (YOLO format).
- Models: `data/weights/`.
- Reports/images: `data/reports/`.

## Principles

Isolation and single responsibility are enforced: each module has one host purpose and tests do not require the heavy ML stacks to run — everything that depends on torch/ultralytics is lazy-imported behind the demo/real switch.

## PNS calibration note

The numeric thresholds in `config_rules.json` are a faithful-but-provisional encoding of PNS/BAFS 98:2022 observable criteria. Confirm against the official standard PDF (linked in Chapter 2 references) before thesis submission and update the JSON — no code change required.

---

# Phase 2 (2026-08-07) — Multi-view ear inspection + coverage-based grading

## Decisions agreed with the thesis authors
- **Capture:** turntable rig, 4 roll views per ear (0°/90°/180°/270°), `ear{id}_v{1..4}` filenames. The resting contact band is never photographed → accepted as a documented delimitation.
- **Annotation:** box every visible defect in every view where it appears (same class per physical spot); box whole-ear defects (deformity, discoloration) with ear-sized boxes; one `corn_ear` box per view. YOLO export, class order fixed to `corn_ear, mold, insect_damage, discoloration, deformity, missing_kernels`.
- **Grading is extent-based (authoritative PNS numbers):** Extra = 0% coverage, Class I < 5%, Class II < 10%, Reject ≥ 10%. Utilization: coverage ≥ 10% → reject; any mold/insect → animal feed; < 5% → human consumption; else food processing.

## New/changed components
- **`views.py`** — per-ear merge: variety majority vote (ties → confidence), per-class defect dedup via roll-adjacency + size-similarity (union-find; fallback per-view max when a view has >1 box), single-largest-area per merged defect, coverage % = Σ largest areas / largest ear-box area, kernel completeness = min, ear size = longest box.
- **`rules/config_rules.json` + `engine.py`** — grade assignment on coverage thresholds; overrides for severe classes; utilization ordering.
- **`pipeline.py` / `report.py` / `main.py`** — `POST /inspect` accepts 1..N files; report carries `views[]`, `defect_coverage`, merged defects with provenance; per-view annotated images + a grid image endpoint.
- **Frontend** — multi-photo upload; per-view thumbnails; merged coverage %, provenance chips.
- **Splitter** — leak-free per-ear grouping (ear identity from filename prefix).
- **`validate_dataset.py`** — label↔image pairing, class range, mandatory `corn_ear`, ear consistency.
- **`eval/eval_merge.py`** — model coverage/grade vs. per-ear ground-truth CSV agreement.
- **Train fix** — separate val transforms (no random flip on val).

## Testing
45 tests: merge truth table (adjacent duplicate → 1, non-adjacent → 2, whole-ear → 100% coverage, tie-breaks, N=1 passthrough), coverage truth table (0 / <5 / <10 / ≥10, severe override), leak-free splits, dataset validation, multi-view API end-to-end.