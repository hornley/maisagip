# Unfilled Tip Defect Class — Design

**Status:** Draft for review

**Date:** 2026-10-20

## Goal

Add `unfilled_tip` as a localized detector defect for the visibly underfilled
tip region of a corn ear. Keep `missing_kernels` for separate kernel gaps
elsewhere, and preserve all existing detector IDs.

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
expectations that currently assert eight classes. Historical documents remain
historical.

The estimated `kernel_completeness` must include the union of
`missing_kernels` and `unfilled_tip` boxes, clipped to the ear box. This keeps
the completeness estimate consistent with the annotation rule that an
`unfilled_tip` box replaces missing-kernel boxes within the tip region and
avoids counting overlapping predictions twice.

## Recommendation integration

Recommendation policy is a separate implementation. This taxonomy change adds
no outlet or safety rule for `unfilled_tip`. The inspection report should pass
the new defect name and updated kernel-completeness trait through the existing
report contract. The defect may affect an outlet indirectly if its coverage
changes the grade under existing rules. Integrate the separate recommendation
implementation without changing its approved policy in this class addition.

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
6. Appearance labels, grading thresholds, recommendation policy, active data,
   archived data, weights, and existing bundles are not otherwise changed.

## Verification scope

Update existing taxonomy, annotation, completeness, generated-metadata, and
report expectations as needed. Verify the scoped diff and generated class
order. Do not train a detector or regenerate user bundles in this change.

## Open integration decision

The separate recommendation work still needs a confirmed outlet for Rejects
without visible mold or decay. The farm interview mentions goat feed or cooking
for clean rejects; current code uses poultry feed. Keep that recommendation
decision out of this taxonomy spec until confirmed.
