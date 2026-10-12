# Mixed Variety Results and Inspection Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show navigable per-view results for confident mixed-variety uploads and simplify successful inspection summaries and eligible outlet recommendations.

**Architecture:** Preserve the existing mixed-variety rejection and enrich only that HTTP 400 response with JSON-safe per-view results. The frontend pairs those results with the selected upload previews in an accessible modal. Successful report inference remains unchanged; grade explanations, summary text, and outlet recommendations are simplified at their existing backend/frontend boundaries.

**Tech Stack:** Python, FastAPI, pytest, vanilla JavaScript, HTML, CSS.

---

## File map

- `backend/app/pipeline.py` — attach per-view predictions and detections to the confident mixed-variety exception.
- `backend/app/main.py` — return structured mismatch data while preserving the current `detail` text and HTTP 400 status.
- `backend/tests/test_pipeline.py` — assert the mismatch response includes ordered per-view variety and defect data.
- `frontend/index.html` — add modal markup for per-view mismatch results.
- `frontend/app.js` — validate and render mismatch payloads, manage preview URLs and modal navigation, and simplify normal report rendering.
- `frontend/styles.css` — style a responsive, keyboard-accessible mismatch dialog and the simplified recommendations list.
- `backend/app/rules/engine.py` — emit only the assigned grade's met criterion, or a concise Reject outcome.
- `backend/tests/test_rules.py` — assert that failed higher-grade criteria are omitted.
- `backend/app/recommendations.py` — filter outlet options by grade and mold/ear-decay restrictions.
- `backend/tests/test_recommendations.py` — assert exact allowed outlet sets and that ineligible options are omitted.

## Task 1: Return per-view results with confident variety mismatch errors

**Files:**

- Modify: `backend/app/pipeline.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Extend the existing mixed-variety API regression test**

In `test_confident_mixed_variety_views_return_actionable_400`, make the mocked detector return a different visible defect for each view. Keep the two existing confident, different variety predictions. Assert the response remains HTTP 400, retains the human-readable `detail`, and includes results in upload order:

```python
payload = res.json()
assert res.status_code == 400
assert "different corn varieties" in payload["detail"]
assert [view["view"] for view in payload["view_results"]] == [1, 2]
assert [view["variety"]["class"] for view in payload["view_results"]] == [
    "yellow_sweet_corn",
    "white_corn",
]
assert [view["defects"][0]["class"] for view in payload["view_results"]] == [
    "mold",
    "missing_kernels",
]
```

- [ ] **Step 2: Run the focused regression to confirm it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_pipeline.py::test_confident_mixed_variety_views_return_actionable_400 -q`

Expected: the current response has no `view_results` field.

- [ ] **Step 3: Carry JSON-safe view summaries on the exception**

Give `InconsistentInspectionInput` a `view_results` attribute. When `inspect_ear` detects a confident disagreement, construct one result per `per_view` item with one-based `view`, the classifier's `variety`, and that view's already threshold-filtered `detections` under `defects`. Raise the exception with the existing message and this ordered list. Keep the successful report path and low-confidence disagreement path unchanged.

- [ ] **Step 4: Preserve the mismatch text and add structured response data**

In the `InconsistentInspectionInput` branch of `/inspect`, return a 400 JSON response with both fields:

```python
return JSONResponse(
    status_code=400,
    content={"detail": str(exc), "view_results": exc.view_results},
)
```

Import `JSONResponse` if it is not already imported. Keep all other validation errors on their existing response path.

- [ ] **Step 5: Run the focused regression and commit**

Run: `.venv/bin/python -m pytest backend/tests/test_pipeline.py::test_confident_mixed_variety_views_return_actionable_400 -q`

Expected: PASS with two ordered results, including each variety and the mocked defects.

```bash
git add backend/app/pipeline.py backend/app/main.py backend/tests/test_pipeline.py
git commit -m "feat: include per-view results in variety conflicts"
```

## Task 2: Add the per-view mismatch dialog

**Files:**

- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`

- [ ] **Step 1: Add dialog structure to the Inspect page**

Add a hidden modal container with `role="dialog"`, `aria-modal="true"`, and a labelled heading. Include the mismatch explanation, image, selected view number, friendly and canonical variety names, defect list, no-defects message, Previous, Next, and Close controls. Do not add any variety confidence field.

- [ ] **Step 2: Retain selected preview URLs for the dialog**

When `selectFiles(files)` creates each preview URL, keep it in an array parallel to `selectedFiles`. Revoke prior object URLs when a new selection replaces them. Reuse the matching URL for each dialog view; do not upload or embed image bytes a second time.

- [ ] **Step 3: Recognize only valid structured mismatch responses**

In the failed `/inspect` response path, check that `view_results` is an array and that each entry has an integer `view`, a string `variety.class`, and an array `defects`. If valid and aligned with selected uploads, open the modal. Otherwise, use `showError(data.detail || ...)` as today. Clear any prior dialog state when the user selects new files.

- [ ] **Step 4: Render and navigate results accessibly**

Render the current uploaded image, one-based view number, friendly variety name plus readable canonical class, and defect-class counts. Exclude the `corn_ear` detector fallback from the defect list because it marks the ear boundary rather than a defect. Show a clear “No visible defects detected” state when no defect classes remain. Previous and Next update the image and data together and are disabled at the first and last view. Focus the dialog on open, close on Escape, and return focus to `inspectBtn` when dismissed. On close, leave the original mismatch text in the inline error alert.

- [ ] **Step 5: Style the dialog responsively**

Use the existing green, cream, and gold design tokens. Match the approved mockup: image and view thumbnails alongside variety and defects on wide screens; stack the image and details on narrow screens. Keep all controls keyboard-focusable and retain visible focus styling.

- [ ] **Step 6: Commit the dialog slice**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: show per-view mixed variety results"
```

## Task 3: Simplify successful inspection details and grade explanations

**Files:**

- Modify: `frontend/app.js`
- Modify: `backend/app/rules/engine.py`
- Test: `backend/tests/test_rules.py`

- [ ] **Step 1: Update grade-reason expectations**

Keep the passing-grade explanation test. Use 6% coverage to assert the single Class II criterion mentions 10% but contains neither “Fails” nor the failed Class I 5% threshold. Add a 26.43% Reject case that checks for a concise rejection outcome and no `Fails` entries.

- [ ] **Step 2: Emit only the assigned criterion from the grade engine**

In `_assign_grade`, return immediately with one “Meets … criteria” reason when a grade passes; do not append failure strings while evaluating preceding grades. If all classes fail, return a single Reject reason that states the coverage and the Class II cutoff without enumerating failed classes.

- [ ] **Step 3: Simplify merged defect labels and inspection time**

In `render(report)`, render merged defect metadata as the deduplicated count only, such as `×2`; remove the coverage percentage and `views ...` suffix. Format the inspection timestamp using `Intl.DateTimeFormat` with medium date and short time styles, and fall back to the raw timestamp if it cannot be parsed. Keep the existing view count after the formatted timestamp, for example `Inspected Oct 12, 2026, 10:45 AM · 2 views`.

- [ ] **Step 4: Run the focused grade tests and commit**

Run: `.venv/bin/python -m pytest backend/tests/test_rules.py -q`

Expected: PASS; normal grades have only their matching met criterion, and Reject has no failed-class list.

```bash
git add backend/app/rules/engine.py backend/tests/test_rules.py frontend/app.js
git commit -m "refine: simplify inspection summary details"
```

## Task 4: Filter recommendations by grade and defect policy

**Files:**

- Modify: `backend/app/recommendations.py`
- Test: `backend/tests/test_recommendations.py`

- [ ] **Step 1: Replace status-oriented tests with eligibility tests**

Add a helper to `backend/tests/test_recommendations.py` that calls `build_recommendations` with the requested grade and defects and returns the option labels:

```python
def labels(grade, defects=None):
    result = build_recommendations(
        grade,
        "yellow_sweet_corn",
        defects or [],
        {"ear_size": "large", "kernel_completeness": 0.96},
        inspection_date="2026-10-09",
    )
    return {option["label"] for option in result["options"]}

assert labels("ExtraClass") == {
    "Local Market", "Export", "Premium Sale", "Processed/Canned",
}
assert labels("ClassI") == {
    "Local Market", "Premium Sale", "Processed/Canned",
}
assert labels("ClassII") == {"Local Market", "Processed/Canned"}
assert labels("Reject") == set()
```

Add cases showing that either `mold` or `ear_decay` leaves only Local Market for non-Reject grades, and that an unrelated defect does not add outlets excluded by grade. Assert visible options contain no `status`, `reason`, or `required_external_checks` fields. Keep the existing valid-price fixture and assert its `price_estimate` remains on the eligible Local Market option.

- [ ] **Step 2: Implement the eligibility intersection**

In `build_recommendations`, map `ExtraClass`, `ClassI`, and `ClassII` to the allowed channel IDs in the spec. Intersect the configured channel rows with that set. For `Reject`, return no outlet options. When defect names contain `mold` or `ear_decay`, further intersect the eligible set with `{"local_market"}`. Emit only each eligible channel's label; do not emit Candidate, Requires checking, unavailable, reason, or requirement fields.

- [ ] **Step 3: Run recommendation tests and commit**

Run: `.venv/bin/python -m pytest backend/tests/test_recommendations.py -q`

Expected: PASS for all grade and defect combinations, with disallowed outlets absent from `options`.

```bash
git add backend/app/recommendations.py backend/tests/test_recommendations.py
git commit -m "feat: filter advisory outlets by grade and defects"
```

## Task 5: Render only eligible recommendations

**Files:**

- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`

- [ ] **Step 1: Render recommendation labels without status/details**

Change `renderRecommendations` to read the eligible `options` array and render each option label as a simple list item. When an option contains a validated `price_estimate`, preserve its current human-readable price/source line beneath that label. Do not render status badges, explanatory candidate text, “Requires …” checks, or unavailable markers. Hide the whole Advisory recommendations card when the array is missing or empty.

- [ ] **Step 2: Remove unused recommendation detail styles**

Remove or leave dormant only the styles that are no longer used by the Inspect recommendation markup; do not change Dataset page styles.

- [ ] **Step 3: Commit the recommendation display**

```bash
git add frontend/app.js frontend/styles.css
git commit -m "refine: show eligible advisory outlets only"
```

## Task 6: Integrated verification

**Files:**

- Verify: `backend/tests/test_pipeline.py`
- Verify: `backend/tests/test_rules.py`
- Verify: `backend/tests/test_recommendations.py`
- Verify: Inspect page in a browser with the API running

- [ ] **Step 1: Run the focused backend checks**

Run:

```bash
.venv/bin/python -m pytest \
  backend/tests/test_pipeline.py \
  backend/tests/test_rules.py \
  backend/tests/test_recommendations.py -q
```

Expected: PASS; successful inspection remains 200, confident mixed varieties remain 400 with view results, low-confidence disagreement remains inspectable, and grade/outlet rules match the spec.

- [ ] **Step 2: Verify the Inspect flow in the browser**

Check one confident mixed-variety upload with at least two files: dialog opens, each view shows its actual preview, friendly/canonical variety, and correct defect counts; Previous/Next update all fields; variety confidence is absent; Escape and Close dismiss the dialog and leave the mismatch guidance visible. Check a normal successful inspection: human-readable date/time, merged defect count without coverage or view numbers, only its met grade criterion, and only grade/defect-eligible recommendation labels. Check a Reject result shows no failed-grade list and no advisory outlets.

- [ ] **Step 3: Inspect the final diff and report results**

Run:

```bash
git diff --check
git status --short
```

Confirm the changes are limited to the listed backend, frontend, and focused test files. Report the commands run and any browser scenarios that could not be reproduced with the available model providers.
