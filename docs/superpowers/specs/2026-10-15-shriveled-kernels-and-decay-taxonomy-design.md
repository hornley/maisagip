# Shriveled Kernels and Decay Taxonomy — Design

Date: 2026-10-15  
Status: Proposed; awaiting user review

## Goal

Restore `shriveled_kernels` as a localized YOLO defect class while preserving
the separate per-ear `dry_appearance` / `plump_appearance` state. Rename the
existing `ear_decay` class to `decay` without changing its meaning or numeric
class ID. Keep class IDs 0–5 unchanged and append `shriveled_kernels` as ID 7.

## Label definitions

| ID | Canonical class | Meaning |
| --- | --- | --- |
| 0 | `corn_ear` | Whole-ear localization, required in each view. |
| 1 | `mold` | Visible fungal growth. |
| 2 | `insect_damage` | Visible insect damage. |
| 3 | `discoloration` | Abnormal kernel color. |
| 4 | `deformity` | Whole-ear shape anomaly. |
| 5 | `missing_kernels` | Missing or damaged kernel patches. |
| 6 | `decay` | The existing visible decay or rotting of the ear or kernels, formerly named `ear_decay`. |
| 7 | `shriveled_kernels` | Localized groups of visibly shrunken, wrinkled, or dented kernels. |

Annotators draw a tight box around each localized `shriveled_kernels` patch in
every view where it appears. `dry_appearance` describes the overall appearance
of the physical ear. Both labels may apply when each criterion is independently
met. Neither label measures moisture content or guarantees freshness. An ear
with no localized shriveled-kernel box is not automatically plump.

## Architecture and behavior

- `backend/app/config.py` remains the canonical ordered detector taxonomy. Its
  class list becomes IDs 0–5 as before, `decay` at ID 6, and
  `shriveled_kernels` at ID 7. Existing numeric label files retain their class
  meanings because neither numeric ID changes.
- Backend decoding, validation, dataset counts, view merging, and generated
  detector YAML already consume the shared class list. Class 7 therefore flows
  through those paths as an ordinary defect and contributes to defect coverage.
- The Dataset annotation UI mirrors the class list. Add the class-7 label and
  palette color; keep class 6's existing decay color under its new name. Report
  rendering uses the same canonical names and matching colors.
- Update active README and dataset protocol text, the evaluation class list, and
  the generated Colab README so current instructions consistently use the
  eight-class order and the new class-6 name. Historical specs and plans retain
  their original wording as records of earlier decisions.
- `decay` retains the current `ear_decay` behavior. `shriveled_kernels` is an
  ordinary coverage defect and does not become a severe-class override.
- Do not change production grading in this taxonomy work. After a reliable
  appearance prediction is integrated into inspection, apply the appearance
  grade policy below; no manual dry/plump choice is added to each inspection.
- Keep the appearance-state API, stored rows, crop workflow, and classifier
  independent from detector class labels.

## Data and model handling

- Leave `data/archive/legacy_shriveled_kernels/` unchanged and do not copy its
  18 class-7 boxes back into active labels. Leave all active images, labels,
  appearance-state rows, weights, and existing bundle files unchanged.
- The active detector labels initially contain no class-7 examples. The
  annotation UI and generated metadata can represent the class, but training a
  detector that can learn it requires deliberately adding reviewed examples and
  using an eight-class training configuration.
- Newly generated detector metadata uses eight class names and the renamed
  `decay` entry. Existing checkpoints and bundles are not rewritten. Since IDs
  0–7 retain the meanings in this design, existing numeric detector outputs map
  through the updated canonical names; class ID 6 is now reported as `decay`.
- Preserve the generic out-of-range behavior: outputs with IDs outside the
  configured class list remain ignored.

## Future grading policy for dry appearance

The appearance result is currently annotation/training data and is not an input
to production inspection. The separate appearance workflow must pass its
per-ear data-support gate and held-out evaluation before a prediction is used
for automatic grading. Until then, the production grade rules stay unchanged.

When reliable appearance predictions are available to inspection, apply this
precedence:

1. Defect-box coverage of 10% or more is Reject, regardless of appearance
   state.
2. With `dry_appearance`, at least one non-ear detector defect (including
   `shriveled_kernels`) and coverage below 10% yields Class II.
3. With `dry_appearance`, no non-ear detector defects, and coverage below 10%,
   the grade is Class I. Dry appearance alone therefore prevents Extra Class.
4. Without `dry_appearance`, keep the existing coverage thresholds.

The binary dry/plump state does not estimate how much of the ear is affected.
The localized `shriveled_kernels` boxes contribute approximate area through the
existing bounding-box coverage calculation. Pixel-level dry-area measurement
would require segmentation masks or another explicit severity annotation.

This policy does not measure moisture content, and `dry_appearance` predictions
must not be interpreted as a moisture reading.

## Compatibility and scope

The serialized class string changes from `ear_decay` to `decay` in detector
outputs, reports, and newly generated metadata. This is the requested
canonical-name change. Numeric YOLO annotations and class IDs remain stable, so
no active label migration is required. External consumers that compare the old
string name will need to use `decay`.

This work does not restore archived annotations, mutate active data, train or
evaluate a model, regenerate bundles, change appearance-state values, add a
manual appearance input to inspection, or change current production grading.
Future appearance integration is when the dry-appearance grade policy above
will take effect.

## Existing checks to keep aligned

Update existing expectations that currently assert a seven-class taxonomy,
reject class 7 as retired, or expect `ear_decay` as the class-6 output. Their
expected order is `corn_ear`, `mold`, `insect_damage`, `discoloration`,
`deformity`, `missing_kernels`, `decay`, `shriveled_kernels`. This design does
not include running the test suite.

## Alternatives considered

- **Keep the seven-class detector and use only the ear-level appearance state:**
  this omits the requested localized boxes for shriveled-kernel patches.
- **Restore the detector class and automatically reactivate the archived
  boxes:** rejected because those 18 boxes all come from one physical ear and
  should remain outside active training until deliberately reviewed and
  included.
- **Replace `dry_appearance` / `plump_appearance` with the detector class:**
  rejected because localized patches and an overall per-ear state have
  different scopes and can both be useful.

## Acceptance criteria

1. The canonical detector order has `decay` at ID 6 and
   `shriveled_kernels` at ID 7; IDs 0–5 are unchanged.
2. The annotation UI and report use the canonical class names and visible
   colors, and new detector metadata reports eight classes.
3. A localized shriveled-kernel box and a whole-ear appearance state can both
   be recorded independently.
4. The archived 18 boxes and current active labels remain byte-for-byte
   untouched by this work.
5. Current production grade thresholds, severity overrides, and recommendation
   rules remain unchanged until reliable appearance prediction is integrated.
6. When integrated, 10% or more box coverage yields Reject; below 10%, dry
   appearance with any non-ear detector defect yields Class II, while dry
   appearance without other defects yields Class I.
