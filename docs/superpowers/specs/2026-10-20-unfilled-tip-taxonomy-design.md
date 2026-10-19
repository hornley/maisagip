# Unfilled Tip and Defect-Based Recommendations — Design

**Status:** Draft for review

**Date:** 2026-10-20

## Goal

Add `unfilled_tip` as a localized detector defect for the visibly underfilled
tip region of a corn ear, and update defect-based recommendations using the
farm interview. Keep `missing_kernels` for separate kernel gaps elsewhere and
preserve all existing detector IDs.

## Label boundary

- Append `unfilled_tip` as class ID 8. IDs 0–7 keep their current meanings.
- Draw one box around the contiguous underfilled tip region.
- Do not add `missing_kernels` boxes inside the region represented by the
  `unfilled_tip` box. Continue labeling distinct missing-kernel patches outside
  that tip region as `missing_kernels`.
- Do not add an `undeveloped_kernels` class. Use `missing_kernels` for those
  separate visible gaps as already decided.
- Treat `unfilled_tip` as an ordinary defect for existing defect-coverage
  grading. It has no new class-specific severity or rejection override.

## Runtime and dataset behavior

`backend/app/config.py` remains the canonical ordered taxonomy. Append
`unfilled_tip` after `shriveled_kernels`; generic decoding, validation, dataset
counts, view merging, and generated detector YAML should consume the updated
list. Update the annotation UI class list and color, report palette, generated
cloud-bundle metadata and instructions, active taxonomy documentation, and
expectations that currently assert eight classes. Update
`backend/app/recommendations.py` and `frontend/app.js` for the Compost safety
override, plus their existing recommendation expectations. Historical
documents remain historical.

The estimated `kernel_completeness` must include the union of
`missing_kernels` and `unfilled_tip` boxes, clipped to the ear box. This keeps
the completeness estimate consistent with the annotation rule that an
`unfilled_tip` box replaces missing-kernel boxes within the tip region and
avoids counting overlapping predictions twice.

## Defect-based recommendations

Keep grade thresholds and grade assignment unchanged. Apply these outlet rules
to the recommendation layer:

- If `mold` or `decay` is detected, recommend `Compost` regardless of grade.
  Suppress food, market, and animal-feed outlets and omit price estimates.
- Otherwise, preserve the existing grade-to-outlet mapping. A `Reject` result
  recommends `Poultry Feed`, including when `insect_damage` is present.
- `unfilled_tip` has no special safety rule. Its coverage can affect the grade
  under existing thresholds, which can then affect the grade-based outlets.

The farm interview describes moldy or rotten ears as compost and clean rejects
as usable for cooking or goat feed. The user-selected `Poultry Feed` label for
Rejects without visible mold or decay is the system's chosen route; it is not a
claim that the interview specifically endorsed poultry feed. A visual result
does not certify feed or food safety, and an absent mold/decay detection is not
a laboratory safety finding.

The report should expose one authoritative recommendation. Do not show a
conflicting legacy utilization recommendation alongside it. `Compost` has no
price estimate; continue to show a validated price only for an eligible
grade-based outlet.

The backend recommendation result is the source of outlet eligibility. The
frontend must show only `Compost` when `safety_exclusions` contains `mold` or
`decay`, even when the assigned grade's ordinary outlet list would exclude
Compost. Without those safety exclusions, it continues filtering outlets
against the grade.

## Data and model handling

- Do not edit or relabel active dataset files as part of the taxonomy change.
- Leave archived labels, existing weights, and previously generated bundles
  untouched.
- New metadata can represent ID 8, but existing checkpoints cannot predict
  `unfilled_tip`; reviewed examples and a later detector retraining are needed
  before production inference can detect it.
- Do not add a special `unfilled_tip` branch to grading or recommendations.

## Scope and acceptance criteria

1. The detector class order is IDs 0–7 as currently configured, followed by
   `unfilled_tip` at ID 8.
2. Annotators can select and draw `unfilled_tip` boxes; annotation and report
   colors are consistent.
3. Generated detector metadata reports nine classes and gives the
   underfilled-tip labeling instruction.
4. Tip-region `unfilled_tip` boxes replace overlapping `missing_kernels`
   labels; separate gaps elsewhere remain `missing_kernels`.
5. `kernel_completeness` accounts for both classes without double-counting
   overlapping predicted regions; ordinary defect coverage grading remains in
   effect.
6. Mold or decay suppresses all food/feed/market outlets and yields `Compost`
   without changing the assigned grade.
7. A Reject without mold or decay yields `Poultry Feed`, including when
   `insect_damage` is present.
8. The frontend displays `Compost` for mold/decay safety exclusions regardless
   of grade and filters other outlets by grade.
9. A Compost recommendation never includes a price estimate.
10. Appearance labels, grading thresholds, active data, archived data,
    weights, and existing bundles are not otherwise changed.

## Verification scope

Update existing taxonomy, annotation, completeness, generated-metadata,
report, backend recommendation, and frontend recommendation expectations as
needed. Verify the scoped diff and generated class order. Do not train a
detector or regenerate user bundles in this change.
