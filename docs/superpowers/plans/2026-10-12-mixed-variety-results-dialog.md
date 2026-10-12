# Mixed Variety Results and Inspection Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show each confidently mismatched upload's own variety, grade, and defects in a navigable dialog, while applying the approved inspection-summary and recommendation presentation rules.

**Architecture:** Keep the existing confident-variety rejection and structured `view_results` response. Derive each mismatch view's grade using the existing grade thresholds and only that view's detections; add a small normalized JPEG preview to each response item so the browser need not decode HEIC/HEIF. Keep successful-report inference unchanged while simplifying its displayed grade criteria, timestamp, deduplicated defect rows, and eligibility-filtered recommendations.

**Tech Stack:** Python, FastAPI, Pillow, pytest, vanilla JavaScript, Node's built-in test runner, HTML, CSS.

---

## File map

- `backend/app/image_io.py` — add a bounded JPEG preview encoder for already-decoded RGB image data.
- `backend/app/pipeline.py` — attach one independently calculated grade and preview bytes/data to each confident-mismatch view result.
- `backend/app/main.py` — serialize preview data as a JSON-safe data URL while preserving HTTP 400 and the current `detail`.
- `backend/app/recommendations.py` and `backend/app/rules/config_recommendations.json` — enforce eligible recommendation sets and the Reject override.
- `backend/app/rules/engine.py` — return only the assigned grade criterion, or a concise Reject criterion.
- `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` — render normalized previews and inline per-view grade; simplify successful inspection details and the Recommendation card.
- `backend/tests/test_image_io.py`, `backend/tests/test_pipeline.py`, `backend/tests/test_recommendations.py`, `backend/tests/test_rules.py` — cover preview encoding, response data, grade separation, recommendation eligibility, and criterion text.
- `frontend/tests/app.test.cjs` — cover mismatch rendering/navigation, no canonical-class or confidence display, successful result labels/date, and recommendation output.

## Task 1: Return self-contained, independently graded mismatch results

**Files:**

- Modify: `backend/app/image_io.py`
- Modify: `backend/app/pipeline.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_image_io.py`
- Test: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Add failing JPEG-preview tests**

Add tests for `encode_preview_jpeg(image, max_dimension=640, quality=80)` using RGB arrays with landscape, portrait, and small dimensions. Decode each result with Pillow and assert JPEG format, RGB mode, preserved aspect ratio, no dimension over 640 pixels, and unchanged dimensions for already-small input. Add a HEIC fixture via `pytest.importorskip("pillow_heif")`; decode it through the existing `decode_image_bytes` path and assert the preview encoder returns browser-renderable JPEG bytes.

- [ ] **Step 2: Run the new image tests and confirm they fail**

Run: `.venv/bin/python -m pytest backend/tests/test_image_io.py -q`

Expected: the new test fails because `encode_preview_jpeg` does not exist.

- [ ] **Step 3: Implement the bounded preview encoder**

Implement `encode_preview_jpeg` in `backend/app/image_io.py`. Convert decoded RGB pixels to a fresh Pillow RGB image, thumbnail in place using Pillow's aspect-ratio-preserving resampling with a 640-pixel longest-side limit, then save as JPEG at quality 80 without EXIF metadata. Do not modify `normalize_image_bytes` or the original uploaded file.

```python
def encode_preview_jpeg(image: np.ndarray | Image.Image, max_dimension: int = 640,
                       quality: int = 80) -> bytes:
    source = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    preview = source.convert("RGB").copy()
    preview.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output = BytesIO()
    preview.save(output, format="JPEG", quality=quality, exif=b"")
    return output.getvalue()
```

- [ ] **Step 4: Verify preview tests pass**

Run: `.venv/bin/python -m pytest backend/tests/test_image_io.py -q`

Expected: PASS; the HEIC test is skipped only when `pillow_heif` is unavailable.

- [ ] **Step 5: Add failing per-view-grade response assertions**

Extend `test_confident_mixed_variety_views_return_actionable_400` to assert upload-order results contain `view`, `variety`, threshold-filtered `defects`, `grade`, and a JPEG data URL. Use distinct defect boxes/areas so the views receive different grades. Assert each grade equals applying the existing grade rule to that view alone, independent of the other view. For equal coverage, assert adding `mold` does not alter grade assignment because safety rules affect recommendations only.

- [ ] **Step 6: Calculate grades and include normalized previews**

Before raising `InconsistentInspectionInput`, process each individual `per_view` entry in isolation using `merge_views([view], confidence_threshold)`, then call the existing `GradeEngine.evaluate` with that isolated coverage, severe flag, variety confidence, completeness, and defect confidences. Include `grade: engine_result["grade_label"]` (one of `Extra Class`, `Class I`, `Class II`, or `Reject`) in the view result. Encode the corresponding decoded image using `encode_preview_jpeg`; serialize it from the `/inspect` mismatch handler as `preview_data_url: "data:image/jpeg;base64,..."`. Preserve the current error `detail`, status 400, classifier result, and defect detections. Do not change successful reports or merge-based inspection grading.

```python
isolated = merge_views([view], confidence_threshold)
view_grade = _engine.evaluate(
    view["variety"]["confidence"], isolated["defect_coverage"],
    isolated["severe_present"], view["traits"]["kernel_completeness"],
    [d["confidence"] for d in view["detections"] if d["class"] != "corn_ear"],
    confidence_threshold=confidence_threshold,
)
result["grade"] = view_grade["grade_label"]
```

In the HTTP 400 serialization, Base64-encode preview bytes and emit a data URL without putting raw bytes in JSON:

```python
preview_data_url = "data:image/jpeg;base64," + base64.b64encode(preview_bytes).decode("ascii")
```

- [ ] **Step 7: Run focused backend tests and commit**

Run:

```bash
.venv/bin/python -m pytest backend/tests/test_image_io.py backend/tests/test_pipeline.py -q
```

Expected: PASS; each view has a valid renderable JPEG preview and a grade computed from its own defect coverage.

```bash
git add backend/app/image_io.py backend/app/pipeline.py backend/app/main.py backend/tests/test_image_io.py backend/tests/test_pipeline.py
git commit -m "feat: include independent view grades and previews"
```

## Task 2: Render the per-view mismatch dialog as approved

**Files:**

- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`
- Test: `frontend/tests/app.test.cjs`

- [ ] **Step 1: Add failing frontend tests for view grade and image source**

Update structured-response fixtures to include grade and `preview_data_url`. Assert malformed grade or non-JPEG preview data makes the response ineligible for the structured dialog. Assert a valid response displays the matching server-provided preview URL, friendly variety name, class inline beside the variety, defects, and navigation state. Assert the canonical class, a separate “Grade” label, and variety confidence are not rendered. Keep tests for empty defects, thumbnail selection, Previous/Next boundaries, closing, and stale responses.

- [ ] **Step 2: Run the frontend suite to confirm failures**

Run: `node --test frontend/tests/app.test.cjs`

Expected: FAIL on the absent preview/grade checks and current canonical-class markup.

- [ ] **Step 3: Update the dialog markup and renderer**

Remove the canonical-class output element. Add a grade chip beside the friendly variety name within the existing “Detected variety” section; do not add a “Grade” caption. Validate that every result aligns with upload order and has a supported class label and JPEG data URL before opening the structured dialog. Set the main image and thumbnail sources from the response data URL, not `URL.createObjectURL` for the uploaded file. Continue to exclude `corn_ear` from visible defects and preserve close/focus behavior.

```html
<div class="mismatch-variety">
  <span class="mismatch-label">Detected variety</span>
  <div class="mismatch-variety-line">
    <strong id="mismatch-variety-name"></strong>
    <span id="mismatch-grade"></span>
  </div>
</div>
```

- [ ] **Step 4: Run frontend tests and commit**

Run: `node --test frontend/tests/app.test.cjs`

Expected: PASS; each view switches image, variety, inline grade, and defects together, with no canonical label or classifier confidence.

```bash
git add frontend/index.html frontend/app.js frontend/styles.css frontend/tests/app.test.cjs
git commit -m "feat: show per-view grade in mismatch results"
```

## Task 3: Simplify successful inspection details and grade criteria

**Files:**

- Modify: `backend/app/rules/engine.py`
- Modify: `frontend/app.js`
- Test: `backend/tests/test_rules.py`
- Test: `frontend/tests/app.test.cjs`

- [ ] **Step 1: Add exact grade-reason tests**

Update `backend/tests/test_rules.py`: for a passing grade, assert there is exactly one “Meets … criteria” reason; for a Class II result at 6% coverage, assert the reason mentions its 10% criterion and does not mention “Fails” or the failed Class I threshold; for Reject at 26.43%, assert a concise Reject reason and no failed-grade list.

- [ ] **Step 2: Implement met-criterion-only grade reasons**

In `_assign_grade`, return the single passing grade criterion as soon as its threshold passes. If no grade passes, return one Reject reason using the existing Class II cutoff. Keep grade thresholds, coverage, and assigned class unchanged. Each returned reasons list contains exactly one string, such as `Meets Class II criteria: 6% coverage is below 10%.` or `Reject: 26.43% coverage is at or above the Class II cutoff of 10%.`

- [ ] **Step 3: Test and implement human-readable inspection summary labels**

Update frontend tests to check that inspection ISO text such as `2026-10-12T10:45:44` renders as a localized date and short time without the `T` or seconds; invalid date input continues to display its raw string. Assert a deduplicated defect row contains only its class and count (for example `Mold ×2`), with neither coverage percentage nor `views 1`. Keep the separate view count after the inspection timestamp if the current summary includes it.

In `frontend/app.js`, use `Intl.DateTimeFormat` with medium date and short time styles for the inspection instant. Render merged-defect class and deduplicated count only; remove coverage and contributing-view suffixes from those rows.

```js
function formatInspectionDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium", timeStyle: "short",
  }).format(date);
}
```

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest backend/tests/test_rules.py -q
node --test frontend/tests/app.test.cjs
```

Expected: PASS; assigned grades are unchanged, only the met criterion is shown, timestamps are human-readable, and merged-defect rows omit coverage and view lists.

```bash
git add backend/app/rules/engine.py backend/tests/test_rules.py frontend/app.js frontend/tests/app.test.cjs
git commit -m "refine: simplify inspection summary details"
```

## Task 4: Apply the exact recommendation eligibility rules

**Files:**

- Modify: `backend/app/recommendations.py`
- Modify: `backend/app/rules/config_recommendations.json`
- Modify: `backend/tests/test_recommendations.py`

- [ ] **Step 1: Add table-driven recommendation expectations**

Assert these exact label sets:

| Grade / condition | Recommendations |
|---|---|
| Extra Class | Local Market, Export, Premium Sale, Processed/Canned |
| Class I | Local Market, Premium Sale, Processed/Canned |
| Class II | Local Market |
| Reject | Poultry Feed |
| Extra Class, Class I, or Class II with mold | Local Market |
| Extra Class, Class I, or Class II with ear decay | Local Market |
| Reject with mold and ear decay | Poultry Feed |

Also assert ineligible labels are absent (not represented with status fields), and every Reject case includes Poultry Feed. Keep validated price-estimate coverage for an eligible outlet.

- [ ] **Step 2: Run recommendation tests and confirm failures**

Run: `.venv/bin/python -m pytest backend/tests/test_recommendations.py -q`

Expected: FAIL for Class II's current Processed/Canned entry and Reject's current empty list.

- [ ] **Step 3: Implement the policy intersection**

Set the allowed grade channels to the table above. For non-Reject grades, intersect with Local Market when `mold` or `ear_decay` is present. For Reject, always return Poultry Feed, even if either safety defect is detected. Emit only eligible recommendation labels and preserve only valid price estimates belonging to an eligible channel. Do not emit Candidate, Requires checking, unavailable, reason, or external-check status fields.

```python
if grade == "Reject":
    eligible = {"poultry_feed"}
else:
    eligible = GRADE_CHANNELS[grade]
    if {"mold", "ear_decay"} & defect_classes:
        eligible = eligible & {"local_market"}
```

- [ ] **Step 4: Run recommendation tests and commit**

Run: `.venv/bin/python -m pytest backend/tests/test_recommendations.py -q`

Expected: PASS for every grade/defect combination with no failed option shown.

```bash
git add backend/app/recommendations.py backend/app/rules/config_recommendations.json backend/tests/test_recommendations.py
git commit -m "feat: enforce recommendation eligibility rules"
```

## Task 5: Render only recommendations that pass the policy

**Files:**

- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`
- Test: `frontend/tests/app.test.cjs`

- [ ] **Step 1: Add frontend assertions for simple recommendation labels**

Test that the section heading is exactly “Recommendation”; only labels present in `options` render; ineligible options and status/reason/requirement fields are absent; valid price detail remains attached to its eligible label; and the card is hidden when there are no options. Reject is not an empty-options case because backend policy always returns Poultry Feed.

- [ ] **Step 2: Implement the card rendering**

Rename “Advisory recommendations” to “Recommendation.” Render only each supplied eligible option's label and any validated price detail. Do not render advisory prose, status badges, Candidate/Requires checking text, or indicators for omitted options. Hide the card for an empty options list.

```js
recommendationsList.replaceChildren();
for (const option of recommendations.options) {
  const item = document.createElement("li");
  item.textContent = option.label;
  // Append a price detail only when the existing price validator accepts it.
  recommendationsList.appendChild(item);
}
recommendationsCard.classList.toggle("hidden", recommendations.options.length === 0);
```

- [ ] **Step 3: Run frontend tests and commit**

Run: `node --test frontend/tests/app.test.cjs`

Expected: PASS; the card contains only eligible labels and optional validated price details.

```bash
git add frontend/index.html frontend/app.js frontend/styles.css frontend/tests/app.test.cjs
git commit -m "refine: show eligible recommendations only"
```

## Task 6: Integrated verification

**Files:**

- Verify: `backend/tests/test_image_io.py`
- Verify: `backend/tests/test_pipeline.py`
- Verify: `backend/tests/test_rules.py`
- Verify: `backend/tests/test_recommendations.py`
- Verify: `frontend/tests/app.test.cjs`

- [ ] **Step 1: Run all focused backend tests**

Run:

```bash
.venv/bin/python -m pytest backend/tests/test_image_io.py backend/tests/test_pipeline.py backend/tests/test_rules.py backend/tests/test_recommendations.py -q
```

Expected: PASS; existing mismatch guidance/status remains, each result uses only its own view for grading, and all recommendation sets match the spec.

- [ ] **Step 2: Run frontend tests**

Run: `node --test frontend/tests/app.test.cjs`

Expected: PASS; modal, navigation, inline variety/grade, date, deduplicated defects, and Recommendation rendering behave as specified.

- [ ] **Step 3: Review the browser flow when model providers are available**

Inspect a multi-file HEIC mismatch and a JPEG/PNG mismatch. Verify real preview images load, the inline class matches each view, visible defects update with navigation, and Previous/Next boundaries and Close/Escape work. Inspect one successful result and confirm date, defect rows, met criterion, and recommendation list. If representative real model outputs are unavailable, report that limitation and rely on API and frontend tests.

- [ ] **Step 4: Inspect final changes without disturbing unrelated files**

Run:

```bash
git diff --check
git status --short
```

Confirm no pre-existing unrelated working-tree changes were staged or reverted, and report focused test results plus any browser scenarios that could not be exercised.
