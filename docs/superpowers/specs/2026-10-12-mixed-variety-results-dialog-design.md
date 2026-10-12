# Mixed Variety Results Dialog Design

## Goal

When inspection rejects multiple uploaded views for confidently showing different corn varieties, keep the current mismatch guidance and show a dialog where the user can review each uploaded view's predicted variety, independently assigned grade, and detected defects. The user can navigate between views with Previous and Next controls. Do not show variety confidence or a “Grade” label in the dialog; show the grade inline beside the friendly variety name. Ensure previews also render for supported HEIC/HEIF uploads. Also simplify successful inspection results and filter recommendations to only the outlets allowed by the grade and defect rules.

## Current behavior

`backend/app/pipeline.py` classifies and detects defects for every uploaded image, then raises `InconsistentInspectionInput` when confident views disagree. The HTTP 400 response already carries each view's variety and detections, and the frontend displays them in a dialog. However, the dialog currently uses browser object URLs for the original uploads, which can fail for HEIC/HEIF in Chromium-based browsers. The response also lacks per-view grades.

## Design

Extend the mismatch error response with structured per-view results while retaining the existing human-readable `detail`. Each result contains its one-based view number, classifier result, that view's threshold-filtered defect detections, independently computed grade, and a JPEG preview generated from the decoded image. Resize the preview so its longest dimension is at most 640 pixels and encode at JPEG quality 80 before Base64 JSON transport. The preview must be usable even when the browser cannot decode the original upload (notably HEIC/HEIF). Do not change the rule that rejects confident cross-view variety conflicts.

On this specific structured mismatch response, the frontend opens an accessible modal over the Inspect page. It keeps the mismatch explanation and single-ear reinspection guidance visible. The modal shows the selected normalized image preview, view number, friendly variety display name, that view's grade shown inline beside the variety name, and a concise list of detected defect classes and counts. Do not show the canonical class as a separate line. When there are no detections, show a clear no-defects message. Previous and Next buttons navigate through all submitted views; they disable at the ends. Closing the dialog returns to the existing inline mismatch message so the user retains the guidance. Do not show classifier confidence, assign a combined grade, or present merged defects for a mixed-variety submission. Compute each view's grade independently by applying the existing grade thresholds to defect coverage from only that view's detections; a different view must not affect it. Safety exclusions continue to govern recommendations, not grade assignment.

Use the existing variety display-name mapping and the existing Inspect page visual language. Keep modal content usable on narrow screens and keyboard accessible: focus enters the dialog, Escape closes it, and focus returns to the triggering context. A normal error without structured mismatch results continues to use the existing inline alert.

For successful inspections, show only the defect class and deduplicated count in each merged-defect row; remove the coverage percentage and view-number list. Format the inspection timestamp as a localized, human-readable date and time without the ISO `T` separator or seconds. Grade explanations show the criterion that was met for the assigned grade and omit failed higher-grade criteria. For Reject, show a concise rejection outcome without listing each failed grade criterion.

The card heading is “Recommendation.” List only eligible outlet names, with no Candidate/Requires checking/unavailable labels and no explanatory status or requirement text. Omit outlets that fail the grade or defect policy. Extra Class allows Local Market, Export, Premium Sale, and Processed/Canned. Class I allows Local Market, Premium Sale, and Processed/Canned, but not Export. Class II allows Local Market only; Export, Premium Sale, and Processed/Canned are excluded. If mold or ear decay is present for a non-Reject grade, Local Market is the only eligible outlet. Reject always shows Poultry Feed, including when mold or ear decay is present. Preserve a validated price estimate for an eligible recommendation when one is available.

## Data flow

1. The pipeline completes per-view classification and defect detection as it does today.
2. On a confident disagreement, the exception carries JSON-safe per-view result summaries, per-view grade data, and bounded JPEG thumbnail bytes encoded for JSON transport.
3. The `/inspect` handler returns HTTP 400 with the existing `detail` and a `view_results` array.
4. The frontend recognizes the structured mismatch response and opens the dialog using the server-normalized per-view previews, avoiding direct dependence on browser support for the original file format.
5. Successful inspection rendering uses the report's grade, defects, inspection timestamp, variety, and recommendations to display the simplified result view.

## Error handling

Each result is aligned by its one-based view number and upload order. Empty defect arrays are valid. The UI falls back to the ordinary error alert if the structured results are absent or malformed; it must not show stale results from a prior inspection. The dialog resets when the user selects a new set of files.

## Verification

- Pipeline/API coverage confirms a confident mixed-variety response contains the original guidance plus one result per uploaded view, including variety, detections, independent grade, and renderable bounded previews.
- Frontend behavior confirms mixed-variety errors open the dialog, navigation updates the image and result details, the grade appears inline beside the variety without a “Grade” label or canonical class, close restores the inline message, and variety confidence is absent.
- Existing behavior remains intact for successful inspection, ordinary validation errors, single-image inspection, and low-confidence variety disagreement.

## Scope

This feature does not create a combined grade for mixed varieties, alter confidence thresholds, change model output classes, or persist rejected inspections. Per-view mismatch grades reuse existing grade thresholds but calculate defect coverage from that view alone. Successful reports keep their existing inference and grade calculations; only their display and recommendation filtering change.
