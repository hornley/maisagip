# Mixed Variety Results Dialog Design

## Goal

When inspection rejects multiple uploaded views for confidently showing different corn varieties, keep the current mismatch guidance and show a dialog where the user can review each uploaded view's predicted variety and detected defects. The user can navigate between views with Previous and Next controls. Do not show variety confidence in the dialog. Also simplify successful inspection results and filter advisory recommendations to only outlets allowed by the grade and defect rules.

## Current behavior

`backend/app/pipeline.py` classifies and detects defects for every uploaded image, then raises `InconsistentInspectionInput` when confident views disagree. `backend/app/main.py` turns that exception into an HTTP 400 response containing only a text `detail`. The frontend therefore cannot display the already-computed per-view results. Uploaded image previews remain available in `frontend/app.js` while the request is in flight.

## Design

Extend the mismatch error response with structured per-view results while retaining the existing human-readable `detail`. Each result contains its one-based view number, classifier result, and that view's threshold-filtered defect detections. Do not change the rule that rejects confident cross-view variety conflicts.

On this specific structured mismatch response, the frontend opens an accessible modal over the Inspect page. It keeps the mismatch explanation and single-ear reinspection guidance visible. The modal shows the selected uploaded image, view number, friendly variety display name (with the canonical class available as supporting text), and a concise list of detected defect classes and counts. When there are no detections, show a clear no-defects message. Previous and Next buttons navigate through all submitted views; they disable at the ends. Closing the dialog returns to the existing inline mismatch message so the user retains the guidance. Do not show classifier confidence, assign a combined grade, or present merged defects for a mixed-variety submission.

Use the existing variety display-name mapping and the existing Inspect page visual language. Keep modal content usable on narrow screens and keyboard accessible: focus enters the dialog, Escape closes it, and focus returns to the triggering context. A normal error without structured mismatch results continues to use the existing inline alert.

For successful inspections, show only the defect class and deduplicated count in each merged-defect row; remove the coverage percentage and view-number list. Format the inspection timestamp as a localized, human-readable date and time without the ISO `T` separator or seconds. Grade explanations show the criterion that was met for the assigned grade and omit failed higher-grade criteria. For Reject, show a concise rejection outcome without listing each failed grade criterion.

The Advisory recommendations card lists only eligible outlet names, with no Candidate/Requires checking/unavailable labels and no explanatory status or requirement text. Do not list an outlet that fails the grade or defect policy. The requested grade policy is: Extra Class allows Local Market, Export, Premium Sale, and Processed/Canned; Class I allows Local Market, Premium Sale, and Processed/Canned; Class II allows Local Market and Processed/Canned. If mold or ear decay is present, Local Market is the only eligible outlet. Reject has no listed outlets.

## Data flow

1. The pipeline completes per-view classification and defect detection as it does today.
2. On a confident disagreement, the exception carries JSON-safe per-view result summaries.
3. The `/inspect` handler returns HTTP 400 with the existing `detail` and a `view_results` array.
4. The frontend recognizes the structured mismatch response, pairs each result with its selected upload preview by index, and opens the dialog.
5. Successful inspection rendering uses the report's grade, defects, inspection timestamp, variety, and recommendations to display the simplified result view.

## Error handling

Each result is aligned by its one-based view number and upload order. Empty defect arrays are valid. The UI falls back to the ordinary error alert if the structured results are absent or malformed; it must not show stale results from a prior inspection. The dialog resets when the user selects a new set of files.

## Verification

- Pipeline/API coverage confirms a confident mixed-variety response contains the original guidance plus one result per uploaded view, including variety and detections.
- Frontend behavior confirms mixed-variety errors open the dialog, navigation updates the image and result details, close restores the inline message, and variety confidence is absent.
- Existing behavior remains intact for successful inspection, ordinary validation errors, single-image inspection, and low-confidence variety disagreement.

## Scope

This feature does not create a combined grade for mixed varieties, alter confidence thresholds, change model output classes, or persist rejected inspections. Successful reports keep their existing inference and grade calculations; only their display and advisory outlet filtering change.
