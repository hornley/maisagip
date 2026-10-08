# Colab YOLO Training Bundle with Fixed Ear-Level Split

## Goal

Produce a reproducible Colab-ready ZIP for the four-model Ultralytics YOLO
comparison. The detector data must keep every view and label for an ear in one
split, preserve the current allocation for ears 1–11, place the specified
reject ears explicitly, and split only ears 16–31 with the existing seeded
80/10/10 policy.

## Required allocation

The split is expressed in YOLO terms as `train`, `val` (validation/evaluation),
and `test`:

| Ear group | Train | Val | Test |
| --- | --- | --- | --- |
| Existing ears 1–11 | `ear003`–`ear010` | `ear001` | `ear002`, `ear011` |
| Fixed ears 12–15 | `ear012`, `ear014` | `ear013` | `ear015` |
| Remaining ears 16–31, seed 42 | 12 ears | 2 ears | 2 ears |

With the current splitter's sorted input order and seed `42`, the remaining
ears are:

- Train: `ear017`, `ear018`, `ear021`, `ear022`, `ear023`, `ear024`,
  `ear025`, `ear026`, `ear028`, `ear029`, `ear030`, `ear031`
- Val: `ear020`, `ear027`
- Test: `ear016`, `ear019`

The final total is 22 train ears, 4 validation ears, and 5 test ears. If all
ears have four views, this is 88/16/20 images, or 124 images overall.

## Design

### Split implementation

Extend the detector split path with a named fixed-split profile rather than
relying on manual file moves. The profile assigns ears 1–15 explicitly and
uses the existing seeded group splitter for all other ears. It must:

1. clear and rebuild only the generated detector split directories;
2. copy images and same-stem YOLO labels together;
3. reject duplicate ear assignments, missing source directories, and invalid
   image/label pairing according to existing validation behavior;
4. keep all views of an ear together; and
5. print the resulting ear IDs and image counts per split so the allocation is
   auditable before training.

The general-purpose random split behavior remains available for existing local
workflows. The fixed profile is selected explicitly by the cloud-preparation
workflow so future dataset additions do not silently change the intended
research split.

### Cloud bundle and Colab flow

Regenerate the cloud ZIP from the prepared detector split. The archive will
contain the prepared `data/detector` images, labels, and YAML plus the minimal
training modules and `requirements-cloud.txt`. Its README will document the
fixed ear allocation and use the following flow:

1. extract the bundle and change into its root;
2. install the pinned/cloud ML dependencies;
3. relocate the YAML `path:` entry to the Colab extraction directory;
4. run `training.yolo_comparison` for `yolov8n`, `yolov8s`, `yolov8m`, and
   `yolo11s` with the shared seed and hyperparameters; and
5. collect model checkpoints, metrics, and plots from
   `data/comparisons/yolo/`.

The test split is not used for training. The comparison runner continues to
train on `train`, use `val` during training, and evaluate the held-out `test`
split after training.

### Verification

Add or update tests for the fixed profile to verify:

- the exact ear-to-split mapping above;
- no ear appears in more than one split;
- each image has its same-stem label in the same split;
- rerunning the profile removes stale generated split files; and
- the cloud ZIP contains the expected split paths, YAML, requirements, and
  Colab instructions without raw HEIC files or local model outputs.

Run the focused splitter, cloud-bundle, and YOLO-comparison tests, then inspect
the ZIP contents and verify the final YAML paths before delivery.

## Scope and limitations

This is an exploratory comparison, not a statistically strong benchmark: the
test set contains only five ears and the four views of each ear are correlated.
The output should report per-model precision, recall, mAP50, mAP50–95, runtime,
and model size where available, and should retain the fixed split for a fair
same-data comparison. More ears and repeated seeds or grouped cross-validation
are needed before claiming a general model ranking.
