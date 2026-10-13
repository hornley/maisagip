# Whole-Ear Dry/Plump Appearance State — Design

Date: 2026-10-13
Status: Approved design; awaiting written-spec review before implementation planning

## Problem

The detector currently represents kernel shriveling with small `shriveled_kernels`
boxes. The current raw dataset has 47 physical ears (188 views), and all 18 boxes
for that class are concentrated on the four views of `ear001`. That is not enough
independent examples to train or evaluate a reliable defect detector. It also
does not match the intended task: classify the visible appearance of an entire
ear as dry or plump, rather than locating individual kernels.

The existing ear box is useful localization. The ear-level appearance state is a
separate label and must not be encoded as another YOLO object class.

## Goal

Keep the existing `corn_ear` box and the remaining detector defects. Replace the
active `shriveled_kernels` box label with an ear-level appearance state that can
be recorded in the Dataset page and used by an independent crop-classification
training/evaluation path when enough independent labeled ears exist.

This work does not train a model, change production inference, or claim that the
current data can reliably distinguish dry from plump ears.

## Dataset evidence and initial state

- `backend/app/config.py` and `frontend/dataset.js` currently list eight detector
  classes, with `shriveled_kernels` as class ID 7. IDs 0–6 are already used by
  `corn_ear`, `mold`, `insect_damage`, `discoloration`, `deformity`,
  `missing_kernels`, and `ear_decay`.
- `data/raw/detector/labels/ear001_v1.txt` through `ear001_v4.txt` contain all
  18 current class-7 boxes. No other ear has a class-7 box.
- The current raw set has 16 white-corn ears and 31 yellow-sweet-corn ears. The
  detector labels contain 188 `corn_ear` boxes. Current defect counts are:
  `mold` 0, `insect_damage` 3, `discoloration` 10, `deformity` 35,
  `missing_kernels` 98, `ear_decay` 14, and `shriveled_kernels` 18.
- `data/raw/ground_truth.csv` is grade-oriented and currently contains only its
  header. Appearance labels will live in a separate per-ear file so an
  appearance-only annotation does not create an incomplete grade record.
- `training/make_dataset_split.py` groups views by physical ear. Its documented
  Colab fixed profile puts `ear001` in validation. The appearance pipeline must
  also group by ear and must not count its four views as four independent ears.

The initial appearance-state file will contain one record per ear. Set
`ear001` to `dry_appearance`, reflecting the user's identification of it as the
dry candidate; leave the other 46 ears unassigned. Absence of a shriveling box
must never be interpreted as `plump_appearance`.

## Label definitions

The persisted field is `appearance_state`. Its allowed values are:

| Stored value | Dataset-page label | Meaning |
| --- | --- | --- |
| `dry_appearance` | Dry appearance | The visible kernels across the ear show clear, repeated wrinkling, shriveling, or denting consistent with a dry/over-mature appearance. |
| `plump_appearance` | Plump appearance | The visible kernels are generally full and rounded, without clear, repeated shriveling. |
| blank | Unassigned | The evidence is insufficient, views disagree, image quality is poor, or no annotator has made a decision. Exclude it from supervised training and evaluation. |

These are visual appearance labels, not measured moisture content, harvest age,
or a guarantee of freshness. Dry silk at the tip alone is not evidence of a dry
ear. This is consistent with the farm interview's description of quality kernels
as plump, shiny, bright, evenly colored, and tight-rowed, and over-mature kernels
as dull, dented, or doughy with dry husk. The interview is
`/Users/hornley/Thesis/CapulongFarms_SweetCorn_Interview_Responses-rev.1.docx`.

If an ear has mixed or ambiguous kernel appearance, leave it unassigned rather
than forcing one of the two training classes. An annotator assigns the state
once per physical ear; all of its views share that state.

## Proposed architecture and data flow

### Existing detection and annotation

- Keep the mandatory whole-ear `corn_ear` box in each view and keep detector
  classes 0–6 unchanged. Do not add dry/plump to `DEFECT_CLASSES`; do not draw
  kernel-level boxes for this appearance task.
- Before changing active YOLO labels, copy the four original `ear001` label
  files containing class 7 into a non-training archive at
  `data/archive/legacy_shriveled_kernels/`. Remove only class-7 lines from the
  active four label files; preserve each file's `corn_ear` and other defect
  lines.
- Remove `shriveled_kernels` from the canonical backend class list and the
  mirrored Dataset-page class list, palette, and help text. Since it is the last
  class, IDs 0–6 remain stable. Update the dataset protocol and README. Existing
  prepared splits, YAML files, caches, bundles, and checkpoints are not silently
  rewritten or deleted; treat artifacts built with the eight-class schema as
  stale and regenerate them only through the explicit preparation workflow.

### Ear-level annotation

- Add `data/raw/appearance_state.csv` with one row per ear and columns
  `ear_id,appearance_state`.
- Add a state selector to the Dataset page's ear/annotation view. It offers
  **Unassigned**, **Dry appearance**, and **Plump appearance**; changes are
  persisted per ear, not per image, and therefore apply consistently to every
  view. The API validates the ear ID and allowed values and reports the state in
  the ear listing/statistics. New ears default to unassigned.
- Keep appearance state separate from `ground_truth.csv`, which stores grade
  evidence. Do not infer plump labels from missing/removed defect annotations.

### Appearance crop preparation and model scaffolding

- Add a preparation command that joins the per-ear state file to raw images and
  YOLO labels. For each assigned ear/view, it crops the full-resolution image to
  that view's `corn_ear` box, then resizes the crop for the classifier. Require
  exactly one valid `corn_ear` box per view used; report missing, duplicate, or
  invalid boxes rather than silently using a whole-image crop.
- Preserve the source images and existing variety-classifier folders. Store
  prepared appearance crops/manifests under a separate `data/appearance/` tree.
  Unassigned ears are excluded. Train/validation/test assignment is at the
  physical-ear level, stratified by appearance state when class counts permit;
  no views from one ear may cross splits.
- Add a separate appearance classifier entry point and checkpoint, initialized
  from the existing ImageNet-pretrained EfficientNetV2-S pattern. Start with the
  configured 448-pixel input (`IMAGE_TARGET_SIZE`) after cropping the ear, while
  preserving the option to raise crop resolution if a pixel-level audit shows
  kernel texture is lost. Keep the current variety classifier, its checkpoint,
  and production model selection unchanged.
- Add an evaluator that reports per-view diagnostics and ear-level results by
  aggregating the view probabilities for each physical ear. It emits confusion
  matrix, per-class precision/recall/F1, macro-F1, balanced accuracy, class
  support, and a review file listing each misclassified ear, its views, and
  predicted probabilities. Where the test support permits, add ear-grouped
  confidence intervals. Do not report view-level metrics as independent-ear
  performance.
- Training and evaluation must stop with a clear insufficient-data message if
  the assigned classes are absent from required splits. A split with only one
  dry ear is not a meaningful validation or test set. The current dataset has
  one dry candidate and zero explicitly assigned plump ears, so this scaffold
  must not be used to claim model performance yet.
- Do not connect the untrained appearance checkpoint to the inspection app or
  recommendation rules. Integration is a later decision after a sufficiently
  varied dataset and held-out evaluation support it.

## Validation and tests

Implementation should add coverage for:

1. Appearance-state CSV parsing, allowed values, unknown ear IDs, default
   unassigned state, per-ear persistence, and sharing the state across views.
2. Migration of only class-7 lines from the four affected active label files,
   with byte-identical legacy copies and preservation of class IDs 0–6.
3. Preparation of a crop from the normalized YOLO `corn_ear` coordinates,
   including clear failures for missing, duplicate, or invalid ear boxes.
4. Split integrity: no physical ear in more than one split; unassigned ears are
   excluded; class counts are reported per split; inadequate class support
   prevents training/evaluation from appearing successful.
5. Ear-level aggregation, confusion matrix/metrics, support reporting, and
   misclassification artifact generation.
6. Existing detector, variety-classifier, and Dataset-page behaviors continue
   to work for their remaining classes and labels.

## Alternatives considered

- **Keep `shriveled_kernels` as YOLO boxes:** rejected because the requested
  target is the whole-ear state, and the current class has only one physical
  ear represented.
- **Use whole-ear YOLO boxes labeled dry/plump:** rejected because those boxes
  are localization targets; changing their class conflates object localization
  with an ear-level attribute and would break the existing detector contract.
- **Add kernel-level boxes for this task:** rejected because the intended label
  is an ear characteristic, not separate kernel objects.
- **Store appearance inside the grade CSV:** rejected because appearance is an
  independent target and should be annotatable without entering grading data.

## Out of scope

- Starting training, generating new weights, or asserting model accuracy.
- Automatically labeling the other ears as plump.
- Treating dry silk, variety, lighting, maturity, or image brightness alone as
  the target label.
- Changing grade rules, production recommendations, detector runtime behavior,
  or the existing variety-classifier model.
- Deleting or overwriting existing prepared datasets, bundles, or checkpoints.
