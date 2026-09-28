# YOLO Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible comparison runner for YOLOv7-tiny, YOLOv7, YOLOv7x, YOLOv8n, YOLOv8s, YOLOv8m, and YOLO11s using the same leak-free ear-level dataset split and comparable metrics.

**Architecture:** Keep dataset preparation in the existing splitter and add a comparison module that owns model specifications, training command construction, run isolation, and result summaries. Ultralytics-backed YOLOv8/YOLO11 models run through the installed Python API; YOLOv7 runs through a user-provided official YOLOv7 checkout because its training code and weight format are separate. The runner must never copy comparison weights into the app’s production `data/weights` path.

**Tech Stack:** Python 3.10+, pytest, PyTorch, Ultralytics YOLO 8+, official YOLOv7 repository, JSON/CSV result artifacts.

---

### Task 1: Make the detector data YAML reusable and add comparison model metadata

**Files:**
- Modify: `training/train_detector.py`
- Create: `training/yolo_comparison.py`
- Test: `backend/tests/test_yolo_comparison.py`

- [ ] **Step 1: Write failing tests for model catalog and stable data YAML**

Add tests that assert the catalog contains exactly the requested models in this order:

```python
assert model_names() == [
    "yolov7-tiny", "yolov7", "yolov7x",
    "yolov8n", "yolov8s", "yolov8m", "yolo11s",
]
```

Also assert `build_data_yaml()` writes absolute `path`, `images/{train,val,test}` entries, seven class names, and a trailing newline without touching model output folders.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
python -m pytest backend/tests/test_yolo_comparison.py -q
```

Expected: FAIL because the comparison catalog and reusable YAML behavior do not exist yet.

- [ ] **Step 3: Implement the catalog and reusable data YAML helper**

Define immutable model records with `name`, `family`, and `weights` fields. Use these exact defaults:

```python
YOLO_MODELS = (
    ModelSpec("yolov7-tiny", "yolov7", "yolov7-tiny.pt"),
    ModelSpec("yolov7", "yolov7", "yolov7.pt"),
    ModelSpec("yolov7x", "yolov7", "yolov7x.pt"),
    ModelSpec("yolov8n", "ultralytics", "yolov8n.pt"),
    ModelSpec("yolov8s", "ultralytics", "yolov8s.pt"),
    ModelSpec("yolov8m", "ultralytics", "yolov8m.pt"),
    ModelSpec("yolo11s", "ultralytics", "yolo11s.pt"),
)
```

Move or reuse `build_data_yaml()` from `training/train_detector.py` so both the existing trainer and comparison runner produce the same YAML. Preserve the existing seven-class order from `backend.app.config.DEFECT_CLASSES`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
python -m pytest backend/tests/test_yolo_comparison.py -q
```

Expected: PASS.

### Task 2: Implement isolated training and metric collection for Ultralytics models

**Files:**
- Modify: `training/yolo_comparison.py`
- Modify: `training/train_detector.py`
- Test: `backend/tests/test_yolo_comparison.py`

- [ ] **Step 1: Write failing tests for command configuration and output isolation**

Test that an Ultralytics run receives the requested model, shared `data`, `epochs`, `batch`, `imgsz`, `seed`, and a unique `project/name`, and that result rows have the model name and run directory. Test that production `config.DETECTOR_WEIGHTS` is not written by comparison helpers.

- [ ] **Step 2: Implement the Ultralytics adapter**

Add a callable adapter that lazily imports `ultralytics.YOLO`, creates the model from the spec weight name, and calls:

```python
model.train(
    data=str(data_yaml),
    epochs=epochs,
    batch=batch,
    imgsz=imgsz,
    seed=seed,
    project=str(output_root),
    name=spec.name,
    exist_ok=False,
)
```

After training, call `model.val(data=..., split="test", imgsz=..., project=..., name=f"{spec.name}-test", exist_ok=False)` and serialize `box.map50`, `box.map`, `box.mp`, `box.mr`, and the best-weight path into one JSON row. If a metric is unavailable, record `null` rather than fabricating a value.

- [ ] **Step 3: Add CLI controls and model filtering**

Implement `python -m training.yolo_comparison` options:

```text
--models MODEL[,MODEL...]
--data-root PATH                 default data/detector
--output-root PATH               default data/comparisons/yolo
--epochs INT                     default 100
--batch INT                      default 16
--imgsz INT                      default 640
--seed INT                       default 42
--device VALUE                   optional, forwarded to Ultralytics
--dry-run                        print planned runs without training
```

The default model list is all seven requested models. Unknown names must fail with a helpful list. Each model gets a separate run directory and the runner writes `summary.json` and `summary.csv` only after collecting results.

- [ ] **Step 4: Run tests and syntax checks**

Run:

```bash
python -m pytest backend/tests/test_yolo_comparison.py -q
python -m py_compile training/yolo_comparison.py training/train_detector.py
```

Expected: PASS and no syntax errors.

### Task 3: Add the native YOLOv7 adapter with clear dependency checks

**Files:**
- Modify: `training/yolo_comparison.py`
- Test: `backend/tests/test_yolo_comparison.py`

- [ ] **Step 1: Write failing tests for YOLOv7 command construction**

Test that each v7 variant maps to its native config:

```text
yolov7-tiny -> cfg/training/yolov7-tiny.yaml
yolov7      -> cfg/training/yolov7.yaml
yolov7x     -> cfg/training/yolov7x.yaml
```

Test that missing `--yolov7-root` produces an actionable error explaining that the official YOLOv7 checkout is required, and that no run starts.

- [ ] **Step 2: Implement the YOLOv7 adapter**

Accept `--yolov7-root PATH` and optional `--yolov7-python PATH`. Validate that `train.py`, `test.py`, and the selected config exist. Construct the native `train.py` command with the same data YAML, epochs, batch, image size, seed, and a unique run directory. Use the corresponding local weight file from `--yolov7-weights-dir PATH` when present; otherwise pass the standard weight filename and let the native repository report a missing download clearly.

Run native `test.py` against the held-out `test` split and parse the standard aggregate metrics line when available. Preserve raw stdout/stderr in the model run directory and write `null` metrics with a warning if the checkout’s output format differs.

- [ ] **Step 3: Add dry-run coverage for all seven models**

Ensure dry-run prints one complete command per model, including the v7 checkout requirement in the three v7 commands and the Ultralytics model name in the four Ultralytics commands, without importing heavy ML packages or creating production weights.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest backend/tests/test_yolo_comparison.py -q
```

Expected: PASS.

### Task 4: Document the experiment and preserve the production training path

**Files:**
- Modify: `README.md`
- Modify: `docs/dataset-protocol.md`
- Modify: `training/train_detector.py`
- Test: `backend/tests/test_yolo_comparison.py`

- [ ] **Step 1: Add regression coverage for existing single-model training**

Assert the existing `train_detector` CLI still defaults to `yolo11n.pt`, uses `data/detector/data.yaml`, and remains responsible for copying only its single best model into `data/weights/corn_yolov11n.pt`. Comparison runs must not change this behavior.

- [ ] **Step 2: Update the training documentation**

Document the exact experiment command:

```bash
python -m training.validate_dataset
python -m training.make_dataset_split --seed 42
python -m training.yolo_comparison \
  --epochs 100 --batch 16 --imgsz 640 --seed 42 \
  --yolov7-root /path/to/yolov7
```

Explain that the four views of each ear remain in one split, results are stored under `data/comparisons/yolo/`, YOLOv7 requires its native repository, and the small dataset makes these exploratory results rather than publishable benchmark claims. Include a dry-run command and the single-model filtering example.

- [ ] **Step 3: Run the full verification suite**

Run:

```bash
python -m pytest backend/tests -q
python -m py_compile training/*.py
git diff --check
```

Expected: all existing tests pass, the new comparison tests pass, and there are no whitespace errors.
