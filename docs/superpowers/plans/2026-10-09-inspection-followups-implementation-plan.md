# Maisagip Inspection Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver clearer inspection results, configurable confidence handling, fewer duplicate defect boxes, and conditional use/market recommendations, with annotation and retraining deferred to the final phase.

**Architecture:** Keep canonical model labels and grade thresholds stable. Make view presentation changes in the frontend, send one request-scoped confidence cutoff through `/inspect`, validate cross-view variety agreement in the inspection pipeline, tune detector NMS from labeled validation examples, and add recommendations in a separate rules/config layer. Leave dataset annotation and retraining until all earlier phases are complete.

**Tech Stack:** Vanilla JavaScript/CSS, FastAPI, PyTorch, Ultralytics YOLO, pytest, existing ear-level dataset utilities.

---

## Priority and phase order

Do the low-risk, immediately visible improvements first, then tackle behavior changes and data-dependent work. The estimates are relative implementation effort, not elapsed-time promises.

| Priority | Phase | Effort | Why this order |
|---|---|---:|---|
| P0 | Result labels, view captions, and visible box outlines | Small | Improves readability without changing predictions or grades. |
| P1 | Shared 50–80% confidence slider and mixed-variety validation | Medium | One coherent upload/API behavior change; establishes explicit confidence handling. |
| P2 | Diagnose and tune nearby duplicate boxes | Medium | Requires reviewed validation examples before choosing suppression settings. |
| P3 | Conditional use/market recommendations and sourced price ranges | Large | Adds a separate advisory layer and needs real, dated Philippine rate sources. |
| P4 | Shriveled-kernel annotation, split refresh, and model evaluation | Data-gated; do last | Requires annotation and held-out ear coverage; no training is part of the earlier phases. |

P4 is intentionally included only as a future, gated phase in this plan. Do not start annotation, regenerate splits, or train models while implementing P0–P3.

### File map

- `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`: threshold slider, result labels, view captions, recommendation card, and visible box outlines. Preserve the current `varietyDisplayName()` mappings for Sweet Fortune and Sweet Pearl.
- `backend/app/main.py`, `backend/app/pipeline.py`, `backend/app/models.py`, `backend/app/rules/engine.py`, `backend/app/report.py`: per-request threshold validation and propagation, model filtering, disagreement errors, warning state, and threshold provenance.
- `backend/app/views.py`: retain current majority merge for permitted cases; reject confident mixed-variety inputs before merging.
- `backend/app/recommendations.py` (new) and `backend/app/rules/config_recommendations.json` (new): candidate outlet and conditional-use rules, separate from grade assignment.
- `backend/tests/test_pipeline.py`, `backend/tests/test_models.py`, `backend/tests/test_rules.py`, and `backend/tests/test_recommendations.py` (new): API, model-call, rule, and recommendation behavior.
- `training/train_classifier.py` and `training/train_detector.py`: only in the final phase, to support safe continuation/output paths if current checkpoints are selected for fine-tuning.
- `data/raw/detector/labels/` and `data/raw/`: annotation-only work in the final phase; no implementation phase rewrites these files.

### Task 1: Simplify inspection summary and view captions

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`

- [ ] **Step 1: Remove the summary confidence percentage and ear-size fraction display**

Remove the `#overall-conf` element from the result header and remove its assignment from `render(report)` in `frontend/app.js`. Keep `report.grade.confidence` in the API and backend rules. Change the ear-size assignment to:

```js
document.getElementById("ear-size").textContent = report.traits.ear_size;
```

Expected: the summary shows the categorical ear size only and no overall-confidence percentage.

- [ ] **Step 2: Show completion and per-view completeness**

Update view caption construction to use each `views[].traits.kernel_completeness` value, keep the existing per-view variety display name, and replace “clean” with “completed” only when there are no non-ear defect detections:

```js
const viewCompleteness = pct(v.traits.kernel_completeness);
cap.textContent = `View ${v.view} · ${varietyDisplayName(v.variety.class)} · Kernel completeness ${viewCompleteness}` +
  (defects.length ? ` · ${defects.map((d) => d.class.replace(/_/g, " ")).join(", ")}` : " · completed");
```

Expected: each view shows its own estimated completeness and a view with no non-ear defects says “completed.”

- [ ] **Step 3: Review the result layout in the browser**

Inspect a multi-view result with both defective and no-defect views. Confirm captions fit, defect names remain present, and no percentage was removed from unrelated measures such as defect coverage.

### Task 2: Make defect outlines visible before hover

**Files:**
- Modify: `frontend/styles.css`

- [ ] **Step 1: Give each hotspot a persistent high-contrast outline**

Update `.annotation-hotspot` so it has a thin visible outline at rest, for example:

```css
.annotation-hotspot {
  position: absolute;
  padding: 0;
  border: 1px solid rgba(255, 255, 255, 0.95);
  outline: 1px solid rgba(0, 0, 0, 0.75);
  outline-offset: -1px;
  background: transparent;
  pointer-events: auto;
  cursor: help;
}
```

Retain the stronger `:hover` and `:focus-visible` treatment and tooltip. Adjust contrast using the existing annotated views so outlines remain readable on both pale and dark corn/background regions.

- [ ] **Step 2: Check pointer and keyboard behavior**

Confirm box hit areas remain the same, tooltips still appear on hover/focus, and the outline does not block the image or adjacent hotspot interaction.

### Task 3: Add the shared 50–80% confidence slider

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `backend/app/main.py`
- Modify: `backend/app/pipeline.py`
- Modify: `backend/app/models.py`
- Modify: `backend/app/rules/engine.py`
- Modify: `backend/app/report.py`
- Test: `backend/tests/test_pipeline.py`
- Test: `backend/tests/test_models.py`
- Test: `backend/tests/test_rules.py`

- [ ] **Step 1: Add the slider and value display**

Place this control in the Inspect upload area near the Inspect button:

```html
<label for="confidence-threshold">Confidence threshold <output id="confidence-threshold-value">50%</output></label>
<input id="confidence-threshold" type="range" min="50" max="80" step="5" value="50">
<small>Higher values can hide lower-confidence boxes and change defect coverage and grade. Classifier and detector scores are not calibrated to each other; this is an operational cutoff, not an accuracy guarantee.</small>
```

In `frontend/app.js`, cache the slider and output elements, update the output on the slider's `input` event, and append the selected fraction to every multipart request:

```js
form.append("confidence_threshold", String(Number(confidenceThreshold.value) / 100));
```

Expected: initial value 50%, selectable values 50, 55, 60, 65, 70, 75, and 80; the selected value accompanies the image request.

- [ ] **Step 2: Validate and thread the request-scoped threshold**

Import `Form` in `backend/app/main.py` and add `confidence_threshold: float = Form(0.5, ge=0.5, le=0.8)` to `POST /inspect`, preserving compatibility for clients that omit the field. Pass it by keyword through `inspect_ear(images_bytes, confidence_threshold=confidence_threshold)`. Keep the value request-scoped; do not store it in mutable global configuration.

Update `models.detect_defects(image, confidence_threshold=0.5)` and `RealDetector.predict(image, confidence_threshold=0.5)` so Ultralytics receives `conf=confidence_threshold`. Apply the same cutoff to demo-detector output so behavior remains consistent when real weights are unavailable. Preserve a compatible default on direct/internal callers.

- [ ] **Step 3: Apply classifier confidence handling and record provenance**

Pass the threshold into `GradeEngine.evaluate()` as the classifier low-confidence cutoff; preserve the independent existing defect-confidence warning rule. Add the selected `confidence_threshold` to successful report data in `build_report()` or immediately after it returns. Keep `grade.confidence` in the report for internal processing even though the summary no longer displays it.

Expected: a low-confidence classifier result can trigger the existing reinspection warning at the user-selected cutoff; the report records which cutoff was used.

- [ ] **Step 4: Add backend threshold tests**

Add tests that submit values `0.5` and `0.8` and confirm they reach the pipeline and report; reject values below `0.5`, above `0.8`, and malformed values with HTTP 422; verify both real-detector kwargs and demo filtering use the selected cutoff; and verify classifier reinspection uses the selected cutoff while defect-confidence checks retain their existing rule. Run:

```bash
.venv/bin/python -m pytest backend/tests/test_pipeline.py backend/tests/test_models.py backend/tests/test_rules.py -q
```

Expected: all selected tests pass.

### Task 4: Reject confident mixed-variety uploads

**Files:**
- Modify: `backend/app/pipeline.py`
- Modify: `backend/app/main.py`
- Modify: `frontend/app.js`
- Test: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Detect conflicting confident view classes before majority merging**

After per-view predictions are collected and before `merge_views()`, form the set of classes whose view confidence is at least `confidence_threshold`. If that set contains two or more classes, raise a dedicated `InconsistentInspectionInput` error listing conflicting view numbers and variety labels.

- [ ] **Step 2: Return an actionable client error**

Catch `InconsistentInspectionInput` before the generic `ValueError` handler and return HTTP 400 with a message such as “These images appear to show different corn varieties. Upload views of one ear.” Keep the existing frontend error display; it must show the response detail clearly.

- [ ] **Step 3: Preserve lower-confidence reports and mark reinspection**

If predictions disagree but do not meet the confident-conflict rule, allow the normal report and set a clear reinspection warning for the disagreement. A single-image upload is never classified as a cross-view conflict. The existing low-classifier-confidence warning still applies to single-image reports using the selected cutoff.

- [ ] **Step 4: Cover agreement and conflict cases**

Add pipeline/API tests for: all views agreeing; 3–1 high-confidence conflict; 2–2 high-confidence conflict; one low-confidence outlier that remains a report with a reinspection warning; a single low-confidence image that remains a report; and mixed classes below threshold. Assert the conflicting view numbers and variety names appear in the client error. Run:

```bash
.venv/bin/python -m pytest backend/tests/test_pipeline.py -q
```

Expected: confident conflicts return the documented client error, and other cases follow the specified behavior.

### Task 5: Diagnose and reduce same-image duplicate boxes

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/models.py`
- Modify: `eval/eval_detector.py` or add `eval/diagnose_duplicate_boxes.py`
- Test: `backend/tests/test_models.py`

- [ ] **Step 1: Create a repeatable validation-set audit**

Use annotated validation ears, grouped by ear ID, to compare current predictions and repeated same-class boxes with YOLO label boxes. Inspect representative cases to label each apparent overlap as a duplicate of one visible defect or a distinct nearby defect. Do not tune NMS on held-out test ears. Keep generated diagnostic outputs in an ignored experiment directory.

- [ ] **Step 2: Compare NMS settings**

Evaluate IoU suppression candidates `0.5`, `0.6`, and the current Ultralytics default `0.7` on validation ears. Record per-class precision/recall or mAP plus counts of confirmed duplicates and missed distinct boxes from reviewed examples. Select a setting only if it reduces confirmed same-class duplicates without merging adjacent true defect patches. Add the measured setting as a named detector config constant and pass it explicitly to `model.predict(iou=...)`; do not add center-distance merging without evidence from labels.

- [ ] **Step 3: Add a detector-call regression check and review examples**

Assert the configured NMS IoU value is passed to Ultralytics. Review the same annotated validation examples at the selected confidence slider settings; do not add center-distance merging unless the labeled review proves it is needed.

### Task 6: Add conditional use and market recommendations

**Files:**
- Create: `backend/app/recommendations.py`
- Create: `backend/app/rules/config_recommendations.json`
- Modify: `backend/app/pipeline.py`
- Modify: `backend/app/report.py`
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Test: `backend/tests/test_recommendations.py`

- [ ] **Step 1: Define the advisory schema and separated rule layer**

Create a `build_recommendations(grade, variety, defects, traits, price_data)` function. Keep it separate from `GradeEngine` so grading thresholds stay unchanged. Return candidate options with a label, status (`candidate`, `requires_check`, or `unavailable`), short reason, and any required external checks. Add a small typed/schema-validated price record representation so missing fields cannot silently become zero-valued estimates.

The configured candidate channels are local market, export, premium sale, and food processing/canned products. Base candidate ranking only on existing visual inputs: variety, visible defect classes and coverage, categorical ear size, and kernel completeness.

- [ ] **Step 2: Add conservative rejection guidance**

For a rejected result, include a practical next action and possible non-food outlet only where configured checks allow it. If mold or a configured safety exclusion is present, return a hold/testing recommendation and do not suggest poultry feed. Never claim feed safety from image predictions. Make this advisory the UI's authoritative recommendation; do not display the legacy `grade.utilization` value when it conflicts with the safety-aware recommendation (the legacy field may remain in the API for compatibility).

- [ ] **Step 3: Add price-data schema without fabricated rates**

Represent each rate with market, variety, grade/product basis, currency, unit, low/high PHP-per-kg values, documented adjustment factors for visible defect coverage, size, and completeness, source, effective date, and valid-through date. Do not add sample prices. Use only configured entries whose validity includes the inspection date. When the table has no matching current entry or any required factor is missing, return no estimate and an explanation. Do not calculate total per-ear price because no ear-weight measurement exists. Do not invent rate values or adjustment factors.

- [ ] **Step 4: Add the recommendation card and test decision paths**

Render candidate uses and reasons separately from assigned grade. Replace the existing `#util-value`/`#util-reason` content with the safety-aware advisory, while leaving grade assignment and coverage thresholds untouched. Show price range, market, unit, source, and effective date when available. Add tests for accepted and rejected grades, mold/safety exclusion (including that poultry feed is absent), missing/stale price data, candidate market checks, and stable grade output. Run:

```bash
.venv/bin/python -m pytest backend/tests/test_recommendations.py backend/tests/test_rules.py backend/tests/test_pipeline.py -q
```

Expected: recommendations respect exclusions, missing/stale prices are omitted, and existing grade labels and thresholds are unchanged.

### Task 7: Final phase — annotate shriveled kernels and train/evaluate (deferred)

**Files:**
- Data annotation: `data/raw/detector/labels/`
- Data validation/splits: `training/validate_dataset.py`, `training/make_dataset_split.py`
- Modify if continuation is selected: `training/train_classifier.py`
- Modify to protect outputs: `training/train_detector.py`
- Evaluation: `eval/eval_classifier.py`, `eval/eval_detector.py`

- [ ] **Step 1: Complete the annotation pass (only when P0–P3 are complete and the user starts this phase)**

Add tight `shriveled_kernels` class-7 boxes only where visibly shriveled patches appear. Follow the dataset protocol and keep matching image/label stems. Do not implement auto-annotation. This task is deliberately not part of current implementation work.

- [ ] **Step 2: Validate and inspect split state before regeneration**

When the user begins P4, run `python -m training.validate_dataset`. Before regenerating anything, inspect and explicitly confirm the exact generated split output directory because `training.make_dataset_split` clears generated split folders. Preserve needed outputs before running `python -m training.make_dataset_split --seed 42`. Confirm all views of each ear remain in one split and regenerate the detector YAML from canonical class config. Never edit or remove prepared splits as part of P0–P3.

- [ ] **Step 3: Prepare safe checkpoint continuation paths**

If fine-tuning the current EfficientNetV2-S checkpoint, add an optional `--init-from` argument to `training/train_classifier.py` that loads the provided state dict into the existing two-class model before optimizer creation. Keep pretrained ImageNet initialization as the default when no checkpoint is provided.

Add a separate detector output argument to `training/train_detector.py` so training never writes through a symlink or overwrites the source checkpoint. Save candidate checkpoints under an experiment path and promote one only after validation/test review.

- [ ] **Step 4: Train only when the data supports the target**

Train the detector with class-7 examples only after annotations cover enough distinct ears for a meaningful validation set. Compare continuation from the current detector checkpoint with the current pretrained baseline using the same refreshed split. Fine-tune the classifier only if new or corrected variety images support it; do not merely add epochs to optimize the tiny validation set.

- [ ] **Step 5: Evaluate with ear-level holdout discipline**

Select checkpoints on validation data. Run classifier and detector test evaluation once on held-out ears and record per-class metrics and ear/image support. Do not tune slider or NMS thresholds on the test set. Keep candidate and production weight files out of Git.

## Final integration review

- [ ] Confirm the shared slider value is attached to each request and returned in its report.
- [ ] Confirm canonical variety and defect IDs and grade thresholds remain unchanged.
- [ ] Confirm no market price is displayed without a matching, current source row.
- [ ] Confirm data annotation and training remained the final phase and the held-out test ears were not used for tuning.
- [ ] Confirm P4 was not started during the P0–P3 implementation unless the user separately authorized starting the deferred data/training phase.
