# Ear Appearance States and Kernel Uniformity — Design

Date: 2026-10-16  
Status: Draft for user review

## Goal

Extend per-ear appearance annotation with a `semi_dry_appearance` state and an
independent kernel-uniformity attribute. Keep appearance, kernel arrangement,
and localized detector defects as separate observations. Do not infer dryness
from missing kernels or kernel-row irregularity.

## Annotation schema

The Dataset page will provide two independent selectors, each assigned once per
physical ear and shared across that ear's views:

| Field | Values | Meaning |
| --- | --- | --- |
| `appearance_state` | blank, `plump_appearance`, `semi_dry_appearance`, `dry_appearance` | Overall fullness and visible wrinkling/shrinkage of intact kernels. |
| `kernel_uniformity` | blank, `uniform`, `irregular` | Consistency of intact kernel size/shape and row spacing/alignment. |

Blank means unassigned or not assessable. `kernel_uniformity` is descriptive
metadata and does not affect grading.

## Annotation boundary

- **Plump appearance:** most visible intact kernels look full and rounded.
  Isolated shriveled patches may still be recorded with localized
  `shriveled_kernels` boxes while the ear-level state remains plump.
- **Semi-dry appearance:** moderate, repeated wrinkling, shrinking, or denting
  is visible across much of the ear, while a noticeable share of intact kernels
  still looks full.
- **Dry appearance:** strong, widespread wrinkling, shrinking, or denting
  dominates the visible intact kernels.
- **Kernel uniformity:** judge intact kernel size, shape, and row alignment.
  Ignore gaps caused by `missing_kernels`; use blank when too few intact kernels
  remain to assess consistency. Mark `irregular` for conspicuous, recurring
  variation, not small natural differences.
- **Missing-kernel-heavy ears:** annotate the missing patches with
  `missing_kernels`. Assess appearance from the remaining intact kernels across
  all views. Choose plump if they are mostly full and sufficient evidence
  remains; otherwise leave appearance unassigned. Missingness alone is not
  semi-dry or dry appearance.

These are visual labels, not moisture measurements. The ear-level states are
mutually exclusive; localized `shriveled_kernels` boxes can coexist with any
state when their criteria independently apply.

## Storage and data compatibility

Keep `data/raw/appearance_state.csv` and extend its schema to
`ear_id,appearance_state,kernel_uniformity`. The reader accepts the existing
two-column header and treats kernel uniformity as blank. On the first annotation
write, the writer emits the new header while preserving each ear's existing
appearance value. Reads do not rewrite the data file. New ears start with both
fields unassigned, and deleting an ear removes its metadata row.

The Dataset API returns and updates each field independently. Statistics report
the counts for all appearance states plus unassigned, and for uniform,
irregular, and unassigned kernel uniformity.

## Classifier and grading behavior

- The separate appearance training and evaluation workflow expands to three
  states. Keep its existing per-class support gate of 25 distinct ears, split
  across train, validation, and test. The current data (8 dry, 20 plump, 0
  semi-dry) does not pass this gate; do not train or enable production
  appearance prediction yet.
- Kernel uniformity is annotation metadata only in this implementation. Do not
  add a uniformity classifier or use uniformity in grading.
- Once reliable appearance predictions are connected to inspection, both
  `dry_appearance` and `semi_dry_appearance` use the approved dry grading rule:
  appearance alone yields Class I; any non-ear detector defect below 10% box
  coverage yields Class II; 10% or more yields Reject. Plump or unassigned
  appearance keeps the existing grade thresholds.
- Production grading remains unchanged until that appearance prediction is
  available. Box coverage is an area proxy; measuring the exact dry portion
  would require segmentation or another severity annotation.

## Scope and compatibility

This change does not alter detector class IDs, localized box definitions,
archived class-7 boxes, active detector labels, weights, or existing training
bundles. It adds ear-level metadata and expands appearance training metadata;
no appearance prediction or grading behavior is activated in production.

## Acceptance criteria

1. The Dataset page can independently set appearance state and kernel
   uniformity once per physical ear.
2. Existing two-column appearance-state files remain readable and retain their
   current appearance values when the new field is first saved.
3. Missing-kernel gaps do not imply dry appearance or irregular kernel
   uniformity; insufficient visible evidence leaves the relevant field blank.
4. Semi-dry is a broad, moderate whole-ear pattern. Localized shriveled patches
   remain box annotations and may coexist with plump appearance.
5. Kernel uniformity has no grading effect. Dry and semi-dry share the approved
   future grade policy, which remains inactive until reliable appearance
   prediction is integrated.
6. The three-class appearance workflow validates support but remains gated from
   training and production inference until every class meets its data gate.
