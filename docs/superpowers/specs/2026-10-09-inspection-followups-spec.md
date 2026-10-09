# Maisagip Inspection Follow-ups Specification

**Status:** Approved for planning. No implementation is included in this document change.

## Goal

Improve inspection readability, make confidence handling explicit, reduce duplicate defect boxes, and provide useful use/market guidance while preserving current grade thresholds and the limits of image-only evidence.

## Current system baseline

- `POST /inspect` accepts one or more images, predicts each image independently, then merges the per-view results into one report.
- The view merger selects the majority variety; ties use the most confident view. It does not report variety disagreement.
- The detector uses a fixed 25% confidence cutoff. Bounding-box overlays are interactive hotspots whose outlines are emphasized on hover/focus.
- Per-view traits already include estimated kernel completeness. Ear size has a categorical label and a size fraction. Overall confidence is displayed as a percentage.
- Current grading rules return Extra Class, Class I, Class II, or Reject, plus a utilization recommendation.
- At planning time, the raw dataset contains 188 images from 47 ears and zero `shriveled_kernels` boxes. The prepared split is behind the raw data, and the checked-in detector YAML is stale. Training utilities regenerate the YAML from the canonical class list.
- Images do not measure moisture, aflatoxin, sweetness, weight, or export certification. The application has no market-price source.

## Requirements

### 1. Inspection display

1. Show categorical ear size without its percentage fraction.
2. Remove the overall-confidence percentage from the summary UI; retain the confidence value in the report for internal warnings and other processing.
3. In every view caption, show `completed` when no non-ear defects were detected. Show that view's estimated kernel completeness from `views[].traits.kernel_completeness`.
4. Make defect-box outlines visible before hover or keyboard focus. Hover/focus may emphasize the outline and reveal its tooltip; default styling must not obscure the corn image.

### 2. Shared confidence slider and input consistency

1. Add one confidence slider to the Inspect upload flow. Minimum and initial value: 50%. Maximum: 80%. Step: 5 percentage points. Display the selected value next to the control.
2. Send the selected value with every inspection request, validate it on the server, and include the applied value in successful reports.
3. Apply the same selected value to:
   - the minimum YOLO confidence for returned defect boxes, replacing the current fixed 25% cutoff (and equivalent filtering of demo-detector output so the control behaves consistently without trained weights); and
   - classifier confidence checks for uncertain predictions and cross-view variety disagreement.
4. If different variety classes each have at least one view whose classifier confidence meets the selected threshold, return an invalid-input error that asks for views of a single ear and identifies the conflicting view numbers/classes where practical.
5. If views disagree but do not meet the confident-conflict rule, return the inspection report with a reinspection warning. A single-image inspection cannot be rejected for cross-view disagreement.
6. Include help text explaining that increasing the shared threshold can hide lower-confidence boxes and change defect coverage and grade. Classifier and detector scores are not statistically calibrated to each other; the shared value is an operational cutoff, not a guarantee of accuracy.
7. Keep the current grading thresholds unchanged. The confidence threshold must not rewrite defect labels or dataset values.

### 3. Nearby or overlapping boxes

1. Review representative predictions against labels to distinguish duplicate boxes for one defect from boxes for nearby, distinct defects.
2. Reduce confirmed duplicates using detector NMS settings or another measured suppression method. Do not merge boxes solely because their centers are close.
3. Compare candidate settings on annotated validation ears and document their effect on duplicates and missed distinct defects before selecting a setting.

### 4. Use and market recommendations

1. Keep the existing grade and grade thresholds unchanged. Add recommendation detail in a separate advisory layer using the grade, variety, visible defect classes/coverage, ear size, and kernel completeness.
2. For Reject results, provide a practical next step or possible alternative use rather than only “Reject.” A poultry/feed suggestion is conditional on validated feed-safety rules. If mold or another configured safety exclusion is detected, do not suggest feed; advise holding the lot for qualified inspection/testing. The new safety-aware advisory is authoritative in the UI; do not surface a conflicting legacy utilization suggestion from the grade engine.
3. For non-rejected corn, list candidate outlets/products such as local sale, export, premium sale, and food processing/canned products, with reasons based on available visual traits. Label them as candidates requiring applicable market/quality checks, not certification or guaranteed eligibility.
4. The first price approach uses editable Philippine baseline rates, not live external pricing. A rate entry includes market, variety, grade/product basis, currency, unit, low/high values, documented adjustment factors for visible defect coverage, size, and completeness, source, effective date, and valid-through date. Display estimates as PHP/kg ranges with market, source, effective date, and uncertainty. Do not estimate total per-ear value because the system does not measure ear weight. Omit the estimate when a rate or required adjustment factor is missing or outside its valid dates; do not invent rates or factors.
5. Do not infer moisture, sweetness, aflatoxin, weight, or certification from RGB images. Feed and market advice must remain conditional on any missing checks.

### 5. Dataset annotation and training — final phase

1. Keep annotation and training after the UI, confidence/input handling, box review, and recommendation work.
2. Add visible shriveled-kernel regions as detector class ID 7 in the dataset annotations. This is annotation work; no automatic annotation feature is requested.
3. Before training, validate the raw data and regenerate classifier/detector splits from the raw source, grouped by ear. Do not hand-edit prepared splits. Keep test ears held out.
4. The dataset protocol's detector target is at least 150 boxes per class (300 is the comfortable target); shriveled-kernel boxes must also cover multiple distinct ears. Retrain the detector only when these annotations support training and evaluation. Fine-tune the classifier only if updated variety data justifies it; simply increasing epochs on the current small split is not a goal.
5. Select models using validation results, then evaluate the selected checkpoint once on held-out ears. Report per-class detector metrics and per-variety classifier metrics with ear counts as well as image counts.
6. Keep trained weights out of Git.

## Implementation sequence

1. Update inspection labels, per-view completeness/status, box visibility, and the shared threshold control.
2. Pass the selected threshold through the API, classifier checks, detector inference, conflict handling, and report.
3. Diagnose nearby boxes and tune suppression against annotated validation examples.
4. Add the separate recommendation/advisory layer and editable, sourced Philippine price data.
5. As the final phase, annotate shriveled-kernel examples, refresh ear-level splits, and retrain/evaluate when the data supports it.

## Acceptance criteria

- Ear size is categorical only; overall confidence percentage is absent from the summary; each view shows completion status and estimated kernel completeness.
- Box outlines are visible without hover and remain distinguishable with keyboard focus.
- The slider starts at 50%, permits 50–80% in 5-point steps, and its validated value is applied and recorded.
- The detector respects the chosen confidence cutoff. Confident cross-view variety conflicts return a clear invalid-input error; lower-confidence disagreement stays inspectable with a warning.
- Box tuning reduces confirmed duplicates without collapsing nearby distinct defects on reviewed annotated examples.
- Existing grade thresholds remain unchanged. Recommendations include reasons, safeguards, and dated price provenance; unsupported safety claims and stale/missing prices are not shown.
- Annotation/training is a later, gated phase using regenerated ear-level splits and a held-out test set.

## Out of scope

- Running annotation, retraining, or replacing local model weights during the first implementation phases.
- Live market-price scraping or API integration.
- Declaring official PNS/BAFS, feed-safety, export, or product certification from image predictions.
- Changing canonical variety/defect IDs or the current grade thresholds.
