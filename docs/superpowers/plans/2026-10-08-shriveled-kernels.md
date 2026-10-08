# Shriveled Kernels Defect Class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `shriveled_kernels` as YOLO class ID 7 across configuration, annotation, validation, reports, documentation, and regression tests without changing existing class IDs.

**Architecture:** Keep `backend/app/config.py` as the canonical ordered taxonomy. Existing backend/training code consumes that list generically; only the frontend’s mirrored annotation list, the report palette, and documentation require explicit additions. Add focused tests around class-7 validation, YAML metadata, view merging, and report rendering while preserving current class IDs.

**Tech Stack:** Python, pytest, FastAPI pipeline modules, vanilla JavaScript annotation desk, Markdown/YAML-style dataset metadata.

---

## File map

- Modify `backend/app/config.py`: append the canonical detector class.
- Modify `backend/app/report.py`: assign a stable annotation color.
- Modify `frontend/dataset.js`: append the class and color so the annotation desk exports ID 7 and keyboard shortcut 8.
- Modify `backend/tests/test_validate.py`: exercise class-7 acceptance.
- Modify `backend/tests/test_views.py`: exercise generic merge/report data for the new class.
- Modify `backend/tests/test_yolo_comparison.py`: assert the generated YAML uses the expanded shared taxonomy.
- Modify `README.md`: document the expanded fixed class order and retraining implication.
- Modify `docs/dataset-protocol.md`: update both class tables, fixed-order snippets, annotation guidance, and class-count target wording where needed.

### Task 1: Add the canonical class and focused failing tests

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/tests/test_validate.py`
- Modify: `backend/tests/test_views.py`

- [ ] **Step 1: Add tests for class 7 before implementation.**

In `backend/tests/test_validate.py`, add a sibling of the `ear_decay` test using label line `7 0.3 0.3 0.1 0.1`, and assert no issues. In `backend/tests/test_views.py`, add a test that creates one view with `corn_ear` and `shriveled_kernels`, calls `merge_views`, and asserts the merged entry has class `shriveled_kernels`, count 1, and coverage `0.5`.

- [ ] **Step 2: Run the focused tests and confirm the expected failures.**

Run:

```bash
pytest -q backend/tests/test_validate.py::test_validate_detector_accepts_shriveled_kernels_class backend/tests/test_views.py::test_shriveled_kernels_detection_merges_like_other_defects
```

Expected: the tests fail because class ID 7 is out of range and the new test class is not yet represented by the canonical taxonomy.

- [ ] **Step 3: Append the class without renumbering existing classes.**

In `backend/app/config.py`, change only the list tail to:

```python
    "missing_kernels",
    "ear_decay",
    "shriveled_kernels",
]
```

- [ ] **Step 4: Run the focused tests again.**

Run the same pytest command. Expected: both tests pass; IDs 0–6 retain their previous meanings and class 7 is accepted by the existing range check.

- [ ] **Step 5: Commit the taxonomy and backend regression coverage.**

```bash
git add backend/app/config.py backend/tests/test_validate.py backend/tests/test_views.py
git commit -m "feat: add shriveled kernels detector class"
```

### Task 2: Keep generated detector metadata and report visuals in sync

**Files:**
- Modify: `backend/app/report.py`
- Modify: `backend/tests/test_yolo_comparison.py`

- [ ] **Step 1: Add an explicit shared-taxonomy assertion to the YAML test.**

Extend `test_build_data_yaml_writes_shared_absolute_dataset_definition` with:

```python
assert f"nc: {len(config.DEFECT_CLASSES)}" in content
assert "shriveled_kernels" in content
assert config.DEFECT_CLASSES[-1] == "shriveled_kernels"
```

Keep the existing exact-content assertion so ordering and count remain covered.

- [ ] **Step 2: Run the YAML test before the report change.**

```bash
pytest -q backend/tests/test_yolo_comparison.py::test_build_data_yaml_writes_shared_absolute_dataset_definition
```

Expected: it passes after Task 1 because YAML generation already consumes `config.DEFECT_CLASSES`.

- [ ] **Step 3: Add a stable report palette entry.**

In `_color_for` in `backend/app/report.py`, add:

```python
        "shriveled_kernels": (255, 193, 7),
```

Do not alter existing colors or severity logic.

- [ ] **Step 4: Run report/model-facing tests.**

```bash
pytest -q backend/tests/test_yolo_comparison.py backend/tests/test_views.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit metadata and report support.**

```bash
git add backend/app/report.py backend/tests/test_yolo_comparison.py
git commit -m "feat: render shriveled kernels in detector outputs"
```

### Task 3: Add the class to the annotation desk

**Files:**
- Modify: `frontend/dataset.js`

- [ ] **Step 1: Append the frontend class in the same order as the backend.**

Change `PV_CLASSES` to append `"shriveled_kernels"` after `"ear_decay"`, and add a matching `PV_COLORS.shriveled_kernels` entry. Use `#ffc107` so the canvas, legend, and saved/report views agree.

- [ ] **Step 2: Verify the existing keyboard selection behavior.**

The annotation code maps class index to shortcut as `index + 1`; after appending, class ID 7 must be selected by key `8`. Inspect the existing keydown handler and update only any hard-coded `1–7` help text or class-count bound if present. Do not change the class IDs or label serialization format.

- [ ] **Step 3: Run the repository’s available test suite and inspect the diff.**

```bash
pytest -q
git diff --check HEAD~1
```

Expected: Python tests pass; `git diff --check` reports no whitespace errors. Manually verify that the frontend list has eight entries and the new class is the final entry.

- [ ] **Step 4: Commit the annotation UI change.**

```bash
git add frontend/dataset.js
git commit -m "feat: annotate shriveled kernels boxes"
```

### Task 4: Update user-facing taxonomy documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/dataset-protocol.md`

- [ ] **Step 1: Update the README detector class order.**

Add `7 shriveled_kernels` after `6 ear_decay` in the fixed-order section, and state that the class is appended for compatibility and requires retraining to be predicted by a detector.

- [ ] **Step 2: Update the dataset protocol class tables and snippets.**

Add row 7 to the class definition table and row 7 to the fixed-order code block. Add annotation guidance describing shriveled kernels as visibly dried/shriveled kernel regions and instruct annotators to box the affected patch, not the entire ear unless the defect truly spans the whole ear. Update any “seven classes” or class-count wording to eight where it refers to the detector taxonomy. Keep existing `deformity`/`discoloration` whole-ear guidance unchanged.

- [ ] **Step 3: Review documentation for consistency.**

Run:

```bash
rg -n "DEFECT_CLASSES|ear_decay|missing_kernels|seven|class order|class table" README.md docs/dataset-protocol.md
```

Expected: every fixed-order list includes IDs 0–7 in the same order, and no stale statement claims ID 6 is the final class.

- [ ] **Step 4: Commit the documentation update.**

```bash
git add README.md docs/dataset-protocol.md
git commit -m "docs: document shriveled kernels annotations"
```

### Task 5: Final verification and integration review

**Files:**
- Review all files changed above.

- [ ] **Step 1: Run the complete backend test suite.**

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify the final class order and generated metadata directly.**

```bash
python - <<'PY'
from backend.app import config
assert config.DEFECT_CLASSES == [
    "corn_ear", "mold", "insect_damage", "discoloration",
    "deformity", "missing_kernels", "ear_decay", "shriveled_kernels",
]
print(config.DEFECT_CLASSES)
PY
```

Expected: the printed list has eight entries with `shriveled_kernels` at index 7.

- [ ] **Step 3: Inspect the complete diff for accidental renumbering or unrelated edits.**

```bash
git diff HEAD~4 --stat
git diff HEAD~4 -- backend/app/config.py backend/app/report.py frontend/dataset.js README.md docs/dataset-protocol.md
git status --short
```

Expected: only the intended class additions, tests, and documentation appear; unrelated pre-existing untracked files remain untouched.

- [ ] **Step 4: Record the implementation result in the final response.**

Report that ID 7 was appended compatibly, the annotation shortcut is 8, tests passed, and existing detector weights need retraining to predict the new class.
