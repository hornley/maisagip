# Colab YOLO Fixed Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a Colab-ready YOLO comparison ZIP containing the exact ear-level split approved by the user.

**Architecture:** Add an explicit `colab-fixed` detector split profile to the existing splitter. It assigns ears 1–15 deterministically, applies the existing seeded group split only to ears 16–31, and leaves the existing random splitter unchanged. Rebuild the prepared detector dataset, update the bundle's embedded README and tests, then generate and inspect `maisagip-cloud-training.zip`.

**Tech Stack:** Python 3, `pathlib`, `argparse`, `shutil`, `zipfile`, pytest, Ultralytics-compatible YOLO YAML.

---

## File map

- Modify `training/make_dataset_split.py`: add the named fixed detector profile, its exact ear mapping, deterministic remainder split, CLI option, and auditable summary output.
- Modify `training/prepare_cloud_bundle.py`: document the exact split and Colab commands inside the archive.
- Modify `docs/dataset-protocol.md`: document how to prepare the fixed comparison split and bundle.
- Modify `backend/tests/test_splitter.py`: test the exact fixed mapping, label/image pairing, and stale-file cleanup.
- Create `backend/tests/test_prepare_cloud_bundle.py`: test archive contents, portable YAML, requirements, README instructions, and filtered files.
- Regenerate `data/detector/images/**`, `data/detector/labels/**`, and `maisagip-cloud-training.zip`: these are the delivered prepared dataset and Colab artifact, not source-code logic.

### Task 1: Add failing tests for the fixed detector profile

**Files:**
- Modify: `backend/tests/test_splitter.py`

- [ ] **Step 1: Add imports and fixed-profile test data.**

Import `split_detector_fixed` and `COLAB_FIXED_DETECTOR_SPLITS` from
`training.make_dataset_split`. Use a temporary raw detector containing one
image and matching label for every ear `ear001` through `ear031`, with two
views per ear so tests can prove that views remain grouped without creating a
large fixture.

- [ ] **Step 2: Write the exact mapping test.**

Add a test that calls:

```python
counts = split_detector_fixed(
    image_root,
    label_root,
    tmp_path / "out" / "detector",
    seed=42,
)
```

Assert the returned image counts are `{"train": 44, "val": 8, "test": 10}`
for two views per ear, and assert the ear IDs in each output directory are:

```python
{
    "train": {
        "ear003", "ear004", "ear005", "ear006", "ear007", "ear008",
        "ear009", "ear010", "ear012", "ear014", "ear017", "ear018",
        "ear021", "ear022", "ear023", "ear024", "ear025", "ear026",
        "ear028", "ear029", "ear030", "ear031",
    },
    "val": {"ear001", "ear013", "ear020", "ear027"},
    "test": {"ear002", "ear011", "ear015", "ear016", "ear019"},
}
```

Assert each ear occurs in exactly one split and each copied image has a
same-stem label in the same split.

- [ ] **Step 3: Write the missing-fixed-ear failure test.**

Omit `ear015` from the raw fixture and assert
`split_detector_fixed(..., seed=42)` raises `ValueError` mentioning
`ear015`. This prevents a silently incomplete research split.

- [ ] **Step 4: Write the rerun cleanup test.**

Run the fixed split once, place a stale image and label in one generated split,
run it again, and assert the stale pair is gone while the fixed ear mapping is
unchanged.

- [ ] **Step 5: Run the new tests to confirm they fail.**

Run:

```bash
python -m pytest backend/tests/test_splitter.py -q
```

Expected: the existing tests pass and the new tests fail because the fixed
profile and constant do not yet exist.

### Task 2: Implement the fixed detector split profile

**Files:**
- Modify: `training/make_dataset_split.py`

- [ ] **Step 1: Define the explicit profile mapping.**

Add this module-level constant:

```python
COLAB_FIXED_DETECTOR_SPLITS = {
    "ear001": "val",
    "ear002": "test",
    "ear003": "train",
    "ear004": "train",
    "ear005": "train",
    "ear006": "train",
    "ear007": "train",
    "ear008": "train",
    "ear009": "train",
    "ear010": "train",
    "ear011": "test",
    "ear012": "train",
    "ear013": "val",
    "ear014": "train",
    "ear015": "test",
}
```

Keep the existing `RATIOS`, generic `split_detector`, and cleanup safeguards
unchanged for callers that still want a random 80/10/10 split.

- [ ] **Step 2: Add `split_detector_fixed`.**

Implement `split_detector_fixed(raw_images, raw_labels, out_root, seed=42)` by:

1. validating and clearing the same six generated detector directories as
   `split_detector`;
2. collecting sorted image files and grouping them by `ear_id`;
3. raising `ValueError` when any key in `COLAB_FIXED_DETECTOR_SPLITS` is absent;
4. passing only unassigned ear groups, in sorted ear-ID order, through the
   existing `_split_groups(..., seed)` function;
5. combining those generated assignments with the explicit mapping;
6. copying every image to `images/{train,val,test}` and its same-stem label to
   `labels/{train,val,test}`; and
7. returning image counts by split.

Require each source image to have a matching label for this fixed profile and
raise a clear `ValueError` listing missing label stems. The generic splitter's
existing permissive behavior remains unchanged.

- [ ] **Step 3: Add a split-summary helper.**

Add a small helper that reads generated image directories and returns sorted
ear IDs per split. Use it only for CLI reporting so the profile's output can be
checked before training without changing the count-returning API.

- [ ] **Step 4: Add an explicit CLI profile option.**

Add:

```python
parser.add_argument(
    "--detector-split-profile",
    choices=("random", "colab-fixed"),
    default="random",
    help="Detector split policy; colab-fixed preserves the approved ear IDs.",
)
```

Call `split_detector_fixed` only when `--detector-split-profile colab-fixed` is
selected. Print both image counts and ear IDs. Keep the default command's
random behavior and classifier behavior unchanged.

- [ ] **Step 5: Run the focused tests.**

Run:

```bash
python -m pytest backend/tests/test_splitter.py -q
python -m py_compile training/make_dataset_split.py
```

Expected: all splitter tests pass.

### Task 3: Update the cloud bundle documentation and tests

**Files:**
- Modify: `training/prepare_cloud_bundle.py`
- Create: `backend/tests/test_prepare_cloud_bundle.py`

- [ ] **Step 1: Update the embedded cloud README.**

Document that the bundle already contains the prepared 31-ear split and add
the exact totals:

```text
train: 22 ears / 88 images
val: 4 ears / 16 images
test: 5 ears / 20 images
```

List the fixed assignments for ears 12–15 and say ears 16–31 were seeded with
`42`. Keep the Colab command sequence aligned with the requested commands:

```bash
pip install -r requirements-cloud.txt
python -m training.relocate_yolo_dataset \
  --data-yaml data/detector/data.yaml \
  --dataset-root data/detector \
  --in-place
python -m training.yolo_comparison \
  --models yolov8n,yolov8s,yolov8m,yolo11s \
  --device 0 \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --seed 42
```

Clarify that `val` is used during training and `test` is the held-out final
evaluation set. Do not add raw HEIC files, local weights, or comparison output
folders to the archive.

- [ ] **Step 2: Add an archive-content test.**

Create a minimal temporary project containing all `_REQUIRED_FILES`, a
`data/detector/data.yaml`, paired JPG/TXT files in each split, and decoy HEIC,
`.pt`, and comparison-output files. Call `create_bundle(project, output)` and
assert the ZIP contains the required Python modules, `data/detector/data.yaml`,
`requirements-cloud.txt`, and `CLOUD_README.md`, while excluding decoys.

Read the archived YAML and assert it contains `path: data/detector` and all
three relative split entries. Read the archived README and assert it contains
the fixed assignments, 22/4/5 totals, `--models yolov8n,yolov8s,yolov8m,yolo11s`,
and `--seed 42`.

- [ ] **Step 3: Run the focused bundle tests.**

Run:

```bash
python -m pytest backend/tests/test_prepare_cloud_bundle.py -q
```

Expected: PASS.

### Task 4: Document and generate the prepared dataset

**Files:**
- Modify: `docs/dataset-protocol.md`
- Regenerate: `data/detector/images/**`, `data/detector/labels/**`

- [ ] **Step 1: Document the fixed preparation command.**

Add the explicit local preparation sequence before cloud bundling:

```bash
python -m training.validate_dataset
python -m training.make_dataset_split \
  --detector-split-profile colab-fixed \
  --seed 42
python -m training.prepare_cloud_bundle \
  --output maisagip-cloud-training.zip
```

Include the exact ear groups and the warning that this is an exploratory
comparison with only five test ears.

- [ ] **Step 2: Rebuild the detector split from all raw detector files.**

Run:

```bash
python -m training.make_dataset_split \
  --detector-split-profile colab-fixed \
  --seed 42
```

Expected summary:

```text
train: 88 images, ears ear003 ... ear031 (22 unique ears)
val: 16 images, ears ear001 ear013 ear020 ear027 (4 unique ears)
test: 20 images, ears ear002 ear011 ear015 ear016 ear019 (5 unique ears)
```

Verify with `find`/`rg` that every split has matching image and label stems and
that no ear ID appears in more than one split.

### Task 5: Build, inspect, and verify the final Colab ZIP

**Files:**
- Regenerate: `maisagip-cloud-training.zip`

- [ ] **Step 1: Build the bundle.**

Run:

```bash
python -m training.prepare_cloud_bundle --output maisagip-cloud-training.zip
```

- [ ] **Step 2: Inspect archive paths and YAML.**

Run:

```bash
unzip -l maisagip-cloud-training.zip
unzip -p maisagip-cloud-training.zip data/detector/data.yaml
unzip -p maisagip-cloud-training.zip CLOUD_README.md
```

Confirm the archive has 88 train images, 16 validation images, and 20 test
images, matching labels, the portable YAML path, and the four-model command.

- [ ] **Step 3: Run focused regression tests.**

Run:

```bash
python -m pytest backend/tests/test_splitter.py backend/tests/test_prepare_cloud_bundle.py backend/tests/test_yolo_comparison.py -q
```

Expected: PASS. If unrelated pre-existing tests fail, record the exact failure
without modifying unrelated user changes.

- [ ] **Step 4: Check the final working tree and deliver the ZIP.**

Run:

```bash
git status --short
```

Do not stage or overwrite unrelated existing user changes. Deliver the
absolute path to `maisagip-cloud-training.zip` and the three Colab cells from
`CLOUD_README.md`.

## Self-review

- Spec coverage: fixed ears 1–15, seeded ears 16–31, ear-level grouping,
  paired labels, portable YAML, Colab commands, archive filtering, tests, and
  exploratory limitations are covered by Tasks 1–5.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps.
- Type/API consistency: `split_detector_fixed` returns the same split-count
  shape as `split_detector`; the CLI selects it through
  `--detector-split-profile colab-fixed`; bundle generation remains
  `create_bundle(project_root, output)`.
