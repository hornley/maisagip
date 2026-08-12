# Maisagip Dataset Uploader — Design

Date: 2026-08-12
Branch: `feat/dataset-uploader`

## Problem

Building the training dataset is manual: photos are hand-renamed to `earNNN_vN`,
copied into `data/raw/classifier/{variety}/` and `data/raw/detector/images/`,
LabelImg/CVAT exports are copied into `data/raw/detector/labels/`, and
`data/raw/ground_truth.csv` is edited by hand. Mistakes surface only later, at
`validate_dataset` time.

## Goal

A web UI (single local user, no auth) that replaces the manual file juggling:

1. Upload photos per ear with auto/assigned `earNNN_vN` naming and routing.
2. Upload per-view YOLO `.txt` labels, validated live.
3. Visualize dataset stats and run validation in the browser.
4. Edit ground truth per ear with server-computed grade.

In-browser box drawing is explicitly out of scope (LabelImg/CVAT/Roboflow stay).

## Architecture

Filesystem stays the source of truth (same layout the pipeline expects) — no
database. New pieces:

- `backend/app/dataset_api.py` — `APIRouter` mounted in `main.py` at `/dataset/*`.
- `frontend/dataset.html` + `frontend/dataset.js` — new page, linked from both
  topbars via a shared `.topnav`. `styles.css` extended with table/form/stat styles.
- `training/validate_dataset.py` — extracted `check_label_lines(name, lines, problems)`
  and `view_number(path)` so single-label validation is reusable at upload time.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/dataset/stats` | ear/image/label counts, boxes per class, missing labels, incomplete ears |
| GET | `/dataset/ears` | per-ear table: variety, views, labels present, ground-truth flag |
| GET | `/dataset/validate` | full validator run; `{ok, problems}` |
| POST | `/dataset/ears` | multipart import (see below) |
| GET | `/dataset/ground-truth` | rows from `ground_truth.csv` |
| PUT | `/dataset/ground-truth` | upsert one row; grade auto-computed |
| POST | `/dataset/split` | run `make_dataset_split`; requires `?confirm=true` |

### Upload contract (`POST /dataset/ears`)

Form fields: `ear_id` (`auto` or `earNNN`), `variety`, `images[]` (1–4),
`labels[]` (0–4, optional). View slot `i` pairs image `i` with label `i`.

Validation (all-or-nothing, no partial writes, 400 with reason on failure):

- variety in `VARIETY_CLASSES`; ear id matches `ear\d+`; image extension whitelist;
  Pillow decodability check.
- per-view label: 5 fields, class in range, coords in [0,1], mandatory class 0 —
  via `check_label_lines`.
- duplicate image rejected via sha256 across the whole raw tree and within the
  upload; existing view file for the chosen ear id rejected (no silent overwrite).
- auto id = max existing ear number + 1, zero-padded to 3 digits.

On success: image written to `classifier/{variety}/earNNN_vN.{ext}` and
`detector/images/earNNN_vN.{ext}`; label to `detector/labels/earNNN_vN.txt`.

### Ground truth grade rule (server, matches protocol §7)

- mold or insect count > 0, or coverage ≥ 0.10 → `Reject`
- coverage ≥ 0.05 → `Class II`
- coverage > 0 → `Class I`
- else → `Extra Class`

## Frontend

`dataset.html` has three sections: **Import an ear** (ear-id input with datalist +
"next free", variety select, four v1–v4 slots each with photo + label pickers and
inline status), **Overview** (stat grid + validation problems), **Ears** table with
label-status badges and a Run-validation button, and a **Ground truth** table with
per-row editable inputs, live client-side grade preview, and Save.

## Error handling & idempotence

- All endpoints return HTTP 400 with human-readable detail; upload writes only
  after every check passes.
- Duplicate imports are rejected with the path of the existing file.
- Ears with < 4 views are reported in stats as "incomplete", never as errors.

## Testing

`backend/tests/test_dataset_api.py` uses `TestClient` with `config.DATA_DIR`
monkeypatched to a temp dir. Covers: tree writes, auto-ID increment, duplicate
rejection, label/image mismatch, missing `corn_ear` box, unknown variety,
ground-truth upsert + auto-grade + CSV content, stats/ears after upload,
validator surfacing a missing label, and the split confirm guard.

## Out of scope

In-browser box annotation, auth/multi-user, remote deployment, dataset deletion UI.
