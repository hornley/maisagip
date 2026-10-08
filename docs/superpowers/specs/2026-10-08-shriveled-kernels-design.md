# Shriveled Kernels Defect Class

## Goal

Add `shriveled_kernels` as a fully supported detector defect class across the Maisagip pipeline, while preserving the meaning of all existing YOLO class IDs and existing annotations.

## Compatibility decision

The new class is appended to the existing detector taxonomy:

| ID | Class |
|---:|---|
| 0 | `corn_ear` |
| 1 | `mold` |
| 2 | `insect_damage` |
| 3 | `discoloration` |
| 4 | `deformity` |
| 5 | `missing_kernels` |
| 6 | `ear_decay` |
| 7 | `shriveled_kernels` |

Appending rather than inserting preserves the interpretation of existing label files and model outputs. Existing detector weights remain usable for their original seven classes, but a detector trained with the new class is required to predict `shriveled_kernels`.

## Scope and behavior

- Add `shriveled_kernels` to the backend shared class list.
- Ensure validation accepts class ID 7 and rejects IDs beyond the expanded list.
- Ensure training and comparison data-YAML generation receives the expanded class count and names through the existing config path.
- Add the class to the browser annotation desk and class legend, using keyboard shortcut 8 and a distinct color.
- Preserve existing generic behavior for per-view detections, cross-view deduplication, coverage calculation, report serialization, and annotated-image rendering. The new class participates through the same paths as other non-ear defects.
- Keep severity rules unchanged: `shriveled_kernels` is not severe unless a future rule explicitly adds it.
- Update README and dataset protocol class tables, annotation guidance, and any fixed shortcut/class-count text.

## Data flow

The class name is defined once in `backend/app/config.py`. Backend validation, YOLO YAML generation, and real-detector class decoding consume that ordered list. The frontend annotation desk mirrors the same fixed order so labels exported by the UI remain compatible with the backend and training artifacts. Reports use the detection class name and a palette lookup; adding a palette entry gives the new class a stable visual identity while retaining the generic fallback.

## Testing

Add or update focused tests to verify:

- the canonical class order includes `shriveled_kernels` at ID 7;
- a class-7 YOLO annotation is accepted and an out-of-range class is rejected;
- generated YOLO metadata reports the expanded class count and ordered names;
- a `shriveled_kernels` detection survives view merging and appears in report output/coverage like another ordinary defect;
- existing class IDs and current merge behavior remain unchanged.

## Out of scope

- Retraining or replacing detector weight files.
- Changing grade thresholds or making shriveled kernels severe.
- Introducing a new class-schema service or refactoring the existing config consumers into a generated registry.
