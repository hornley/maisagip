# Local Colab Model Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local `/inspect` use the two Colab-trained checkpoints without committing model binaries.

**Architecture:** Leave backend code unchanged. Preserve the old detector weights, then create relative symlinks from the two fixed `data/weights` paths to `colab_trained_models`. Restart the backend and verify both providers and an inspection request.

**Tech Stack:** Python, FastAPI, PyTorch, Ultralytics, local filesystem.

---

### Task 1: Establish checkpoint and destination state

**Files:**
- Read: `colab_trained_models/best.pt`
- Read: `colab_trained_models/efficientnetv2_s_corn.pt`
- Read: `data/weights/corn_yolov11n.pt`
- Read: `data/weights/efficientnetv2_s_corn.pt` (if present)

- [x] **Step 1: Check exact paths and checkpoint class metadata**

Run: `ls -l colab_trained_models/best.pt colab_trained_models/efficientnetv2_s_corn.pt data/weights` and load YOLO metadata to compare `names` with `backend.app.config.DEFECT_CLASSES`.

Expected: both sources exist, current detector destination is an old regular file, and the eight class names match exactly.

- [x] **Step 2: Confirm working-tree status**

Run: `git status --short`.

Expected: no user changes are modified or removed; model files remain untracked or ignored.

### Task 2: Activate local weights

**Files:**
- Preserve: `data/weights/corn_yolov11n.pt` as `data/weights/corn_yolov11n.pre-colab.pt`
- Create: `data/weights/corn_yolov11n.pt` (symlink)
- Create: `data/weights/efficientnetv2_s_corn.pt` (symlink)

- [x] **Step 1: Move the old detector checkpoint to the backup name**

Run: `mv data/weights/corn_yolov11n.pt data/weights/corn_yolov11n.pre-colab.pt`, but only after checking the backup path does not exist.

Expected: the old 18 MB file remains available under the backup name.

- [x] **Step 2: Create relative symlinks**

Run: `ln -s ../../colab_trained_models/best.pt data/weights/corn_yolov11n.pt` and `ln -s ../../colab_trained_models/efficientnetv2_s_corn.pt data/weights/efficientnetv2_s_corn.pt`.

Expected: both destinations resolve to source checkpoint files.

- [x] **Step 3: Check repository status**

Run: `git status --short`.

Expected: neither model binaries nor symlinks are staged or committed.

### Task 3: Verify real inspection

**Files:**
- Read: `backend/app/models.py`
- Read: `backend/app/main.py`
- Read: a representative image under `data/raw`

- [x] **Step 1: Restart any running backend process**

Restart the local `uvicorn backend.app.main:app --reload --port 8000` process so `backend.app.models._MODELS` is cleared.

- [x] **Step 2: Check real providers**

Run: `GET /health` against the local app or call `backend.app.models.provider_modes()` in a fresh Python process.

Expected: classifier is `real-efficientnetv2s` and detector is the real-model mode, not either demo mode.

- [x] **Step 3: Run one inspection smoke test**

POST one representative corn image to `/inspect` and assert HTTP 200, a report body, and real provider modes.

- [x] **Step 4: Final integrity check**

Run: `git status --short` and `ls -l data/weights`.

Expected: backup exists, both symlinks resolve, and Git has no staged weight files.
