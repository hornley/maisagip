# Maisagip — Dataset Creation Protocol

This document is the step-by-step procedure for building the annotated dataset that trains and evaluates the Maisagip pipeline. Follow it exactly: the training scripts, validator, and evaluator all depend on the naming and layout conventions below.

## 1. Final layout (what the pipeline expects)

```
data/raw/
├── classifier/
│   ├── white_corn/
│   │   └── ear001_v1.jpg … ear001_v4.jpg, ear002_v1.jpg …
│   └── yellow_sweet_corn/
│       └── earNNN_v1.jpg … earNNN_v4.jpg
├── detector/
│   ├── images/
│   │   └── earNNN_v1.jpg … earNNN_v4.jpg     (same photos as classifier)
│   ├── labels/
│   │   └── earNNN_v1.txt … earNNN_v4.txt     (YOLO format, same stem)
│   └── obj.names                             (optional class-list reference)
└── ground_truth.csv                          (per-ear human grading)
```

- The variety folder the photo sits in **is** the classifier label: `white_corn` or `yellow_sweet_corn`.
- The detector uses the identical photo set; images and labels are paired by filename.
- Filename pattern `ear{id}_v{n}` is mandatory — the leak-free splitter groups views by ear using it.

## 2. Class definitions (fixed order — do not reorder)

| Index | Class | Notes |
| --- | --- | --- |
| 0 | `corn_ear` | whole-ear box, mandatory in every label file |
| 1 | `mold` | fungal growth — severe class |
| 2 | `insect_damage` | weevil/borer damage — severe class |
| 3 | `discoloration` | abnormal kernel color |
| 4 | `deformity` | whole-ear shape anomaly (ear-sized box) |
| 5 | `missing_kernels` | missing/damaged kernel patches |
| 6 | `ear_decay` | visible decay or rotting of the ear or kernels |

Use the class table above as the annotation-tool configuration source so export
indices match. If you maintain a local `obj.names` file, its lines must use this
exact order.

## 3. Scale targets

| Component | Minimum | Comfortable |
| --- | --- | --- |
| Ears | 100 | 200–300 |
| Images (4 views/ear) | 400 | 800–1200 |
| Defect boxes per class | ≥150 | ≥300 |
| Classifier images per variety | 150 | 300 |

> Start with ~30 ears to run the full loop end-to-end, then grow. Defect classes are never balanced in nature — deliberately hunt mold and insect-damaged ears so every class appears.

## 4. Capture rig

- **Turntable** (cheap lazy susan / rotating display) on a flat surface.
- **Camera**: phone on a fixed stand, eye-level, ~40 cm from the ear, fixed zoom, fixed exposure, no flash.
- **Scene**: uniform background, diffused indoor light (no harsh shadows, no direct sunlight).
- **Procedure per ear** — 4 shots at exact quarter-turns:
  1. Place ear on turntable (shank pointing left), shoot = `v1` (0°).
  2. Rotate 90° → `v2`; rotate 90° → `v3`; rotate 90° → `v4` (270°).
- The resting contact band is never photographed. This is an accepted, documented delimitation (see design spec Phase 2) — do not try to photograph it.

**Photo QC — retake immediately if any:** tips not fully visible, blur/motion, hand/shadow in frame, camera height or distance changed, exposure changed.

## 5. File organization and renaming

1. Transfer photos from the phone into a temporary folder, ear by ear, in capture order.
2. Rename each batch: `ear001_v1.jpg … ear001_v4.jpg`, `ear002_v1.jpg …` — consecutive IDs, no gaps.
3. Copy the batch into `data/raw/classifier/{variety}/`.
4. Copy the **same files** into `data/raw/detector/images/` (identical names).
5. JPG or PNG only. Keep original resolution; do not resize or crop.

## 6. Annotation (detector boxes)

**Tool**: LabelImg (free, local, YOLO export) or CVAT/Roboflow (export YOLO `.txt`). Configure the tool with the class table above, or with a local `obj.names` file containing that exact order.

Per image:

1. **`corn_ear` box (class 0) — always**: one tight box around the whole visible ear. Drives size/completeness and is validated as mandatory.
2. **One box per visible defect patch**, tight to the patch.
3. **Same physical defect → same class in every view where it appears.** This is what lets the merge dedup the defect across roll views. Changing class between views splits one defect into several.
4. Whole-ear defects — `deformity`, and extensive `discoloration` — get **ear-sized boxes**.
5. `missing_kernels` boxes cover only the missing/damaged kernel patches, not the whole ear.

Export to `data/raw/detector/labels/` with the same filename stem as the image (`ear001_v1.txt`).

## 7. Ground truth CSV (one row per ear)

`data/raw/ground_truth.csv` is pre-created with the header:

```csv
ear_id,variety,coverage,mold_count,insect_count,missing_count,other_count,grade
```

Example row:

```csv
ear001,yellow_sweet_corn,0.03,1,0,0,0,Class I
```

- `coverage`: your human estimate of total defect percentage, 0.00–1.00 (base it on the merged per-defect areas you observed across views).
- `counts`: number of *distinct* physical defects per class (after merging across views), not box totals.
- `grade`: `Extra Class`, `Class I`, `Class II`, or `Below Class II (Reject)`. The CSV stores the last grade using the `Reject` alias, matching the production rules: Extra = 0% coverage, Class I < 5%, Class II < 10%, Below Class II (Reject) ≥ 10% (any mold/insect presence overrides to Reject/animal feed).

This CSV is the reference for `eval/eval_merge.py` — it is how the merge's correctness gets proven.

## 8. Validation → split → train → evaluate

```bash
# 1. Check pairing, class range, mandatory corn_ear, per-ear consistency
python -m training.validate_dataset

# 2. Leak-free per-ear 80/10/10 split (run only when capture+labels are final)
python -m training.make_dataset_split --seed 42

# 3. Train (GPU recommended)
pip install -r requirements-ml.txt
python -m training.train_classifier --epochs 30
python -m training.train_detector --epochs 100

# 4. Evaluate
python -m eval.eval_classifier
python -m eval.eval_detector
python -m eval.eval_merge --images data/raw/detector/images --csv data/raw/ground_truth.csv
```

Weights land in `data/weights/`; the app auto-switches from demo to real providers when they exist.

### YOLO comparison experiment

The comparison uses this exact model set:

- `yolov7-tiny`, `yolov7`, `yolov7x`: official YOLOv7 checkout and its native
  `train.py` / `test.py` scripts.
- `yolov8n`, `yolov8s`, `yolov8m`, `yolo11s`: Ultralytics.

Run the shared preparation and comparison with:

```bash
python -m training.validate_dataset
python -m training.make_dataset_split --seed 42
python -m training.yolo_comparison \
  --epochs 100 --batch 16 --imgsz 640 --seed 42 \
  --yolov7-root /path/to/yolov7
```

For the preliminary Colab comparison, use the approved fixed detector split
before creating the cloud bundle:

```bash
python -m training.validate_dataset
python -m training.make_dataset_split \
  --detector-split-profile colab-fixed \
  --seed 42
python -m training.prepare_cloud_bundle \
  --output maisagip-cloud-training.zip
```

This preserves the existing ear 1–11 allocation, places `ear012` and
`ear014` in train, `ear013` in validation, and `ear015` in test. Ears 16–31
use the grouped seeded split, producing 22 train ears, 4 validation ears, and
5 test ears (88/16/20 images when every ear has four views). The test set is
held out for final evaluation; the comparison is exploratory because it
contains only five test ears.

The splitter groups filenames by ear ID, so all four views of each ear remain in
the same train, validation, or test split. Every model therefore consumes the same
ear-level split. Comparison results and checkpoints are written below
`data/comparisons/yolo/`. The runner may regenerate the shared
`data/detector/data.yaml`, but comparison runs do not modify the source detector
images or labels and never copy checkpoints into production `data/weights/`.

To inspect the planned commands without importing ML packages or training, use:

```bash
python -m training.yolo_comparison \
  --seed 42 --yolov7-root /path/to/yolov7 --dry-run
```

Run a single model with the comma-separated filter (the Ultralytics models do not
need `--yolov7-root`):

```bash
python -m training.yolo_comparison \
  --models yolo11s --epochs 100 --batch 16 --imgsz 640 --seed 42
```

When using YOLOv7, `--yolov7-python PATH` can select the Python executable from its
environment, and `--yolov7-weights-dir PATH` can point to local files such as
`yolov7-tiny.pt`, `yolov7.pt`, and `yolov7x.pt`. The official YOLOv7 `train.py`
lacks a `--seed` option. The runner records the requested seed and
`PYTHONHASHSEED`, warns that YOLOv7's internal seed remains hardcoded, and marks
those results as exploratory; it does not claim identical internal seeding across
the model families.

The current dataset is only 11 ears / 44 views. The comparison is consequently an
exploratory engineering experiment, not a reliable or publishable benchmark. Add
substantially more ears before drawing general model-performance conclusions.

The regular single-model detector command above remains the production path: it
defaults to `yolo11n.pt`, uses `data/detector/data.yaml`, and copies its one best
checkpoint to `data/weights/corn_yolov11n.pt`.

## 9. QC checklist before training

- [ ] `validate_dataset` reports zero problems
- [ ] Spot-check ~10% of images visually — missed defects are the #1 silent error
- [ ] Every defect class appears in enough boxes (≥150/class minimum)
- [ ] Same defect = same class across views (spot-check several ears)
- [ ] No duplicated ears between classifier and detector sets (they are the same files)
- [ ] `ground_truth.csv` has one row per ear, coverage in [0,1], grade spelled exactly

## 10. Time estimates

- Capture: ~2–3 min/ear including setup and QC.
- Annotation: ~1–2 min per view per ear for an experienced annotator.
- 150 ears ≈ 1.5 capture days + 10–15 h of labeling.

## 11. Common mistakes

| Mistake | Consequence |
| --- | --- |
| Renaming views out of rotation order | Merge dedup "adjacent views" logic sees wrong neighbors |
| Same physical defect labeled with different classes per view | One defect counted as several; coverage inflated |
| Missing `corn_ear` box | Validator fails the file; traits unavailable |
| Whole-ear defects boxed as small patches | Coverage underestimated; ear area mistaken |
| Reusing an ear id across ears | Splitter treats them as one ear (leakage) |
| Editing split folders by hand | Leakage between train/val/test |
