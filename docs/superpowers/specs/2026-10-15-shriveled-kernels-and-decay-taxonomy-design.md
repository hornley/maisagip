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
- Keep severity, grading, and recommendation rules unchanged. `decay` retains
  the current `ear_decay` behavior; `shriveled_kernels` is an ordinary coverage
  defect and does not become a severe-class override.
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

## Compatibility and scope

The serialized class string changes from `ear_decay` to `decay` in detector
outputs, reports, and newly generated metadata. This is the requested
canonical-name change. Numeric YOLO annotations and class IDs remain stable, so
no active label migration is required. External consumers that compare the old
string name will need to use `decay`.

This work does not restore archived annotations, mutate active data, train or
evaluate a model, regenerate bundles, change appearance-state values, or alter
grading and recommendations.

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
5. Grade thresholds, severity overrides, and recommendation rules retain
   their current behavior.
