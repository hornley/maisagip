# Maisagip Android App Design

**Date:** 2026-11-17
**Status:** Approved by the user

## Goal

Replace the field inspection web interface with a native Android app that runs corn ear segmentation, variety classification, defect detection, grading, and recommendations on the phone without an internet connection. Keep dataset preparation, annotation, training, and evaluation as developer tools outside the app.

## Approved product decisions

- Use a native Kotlin app with Jetpack Compose.
- Package ONNX exports of the existing three trained models with the app so they are available offline after installation.
- Use a minimalist, harvest-first home screen. Do not ask the user to choose an inspection type.
- Infer the inspection path from the selected media: one image is segmented into one or more ears; multiple images are treated as views of one ear.
- Keep individual results and photos while a harvest is active. Active harvest ear rows open detailed results.
- When starting the next harvest, confirm the cleanup, retain a compact harvest summary, then delete the previous harvest's photos and per-ear rows.
- Keep archived harvest history as summaries only, with no individual ear rows.

## User flow

### Home and capture

The home screen shows the active harvest and a primary action to inspect more corn. If no harvest is active, offer a single action to start one. Keep the copy short and omit introductory or technical text.

Open the Android camera when the user chooses **Take a photo** and the system photo picker when they choose **Choose from gallery**. Request camera permission only for camera use. Copy chosen images into app-private temporary storage so the inspection and active harvest work offline and survive app restarts.

After media selection, show thumbnails and a short automatic interpretation before inspection:

- **One image:** run YOLO11s-seg to find separate ear masks. Analyze each segmented ear independently and create one harvest row per detected ear.
- **Multiple images:** treat the images as views of the same ear and merge the per-view results into one ear result. Four roll views are recommended, but the app does not reject other counts solely for differing from four.

The user confirms the media selection by starting inspection; there is no inspection-type picker. When view variety predictions confidently conflict, stop the merge and let the user review the selected views and replace them. If segmentation finds no usable ears, show a plain recovery message and let the user retake or choose other photos. Do not create an empty or misleading ear result.

### Inspection result

Show the grade first, followed by variety and any reinspection flag. While the harvest is active, show the annotated ear image, defect coverage, kernel completeness, detected defects, grade rationale, and utilization recommendation. Keep the notice that visual findings do not determine food safety.

Use the existing default confidence threshold of 0.50. Do not show a technical threshold control in the primary flow. Preserve the current 0.50–0.80 control in an advanced setting so the web app's tuning capability is not silently lost.

Each completed result is saved to the active harvest automatically. A one-image segmentation result may create several ear rows; a multi-image inspection creates one row. Selecting an active harvest row opens the full result for that ear. The result screen links back to its harvest and offers **Inspect more**.

Low confidence or confident view disagreement must be labeled as requiring review, not presented as certain. If the bundled runtime or any required model fails to load, block inspection with a clear error. Never silently use a demo heuristic in the released app.

### Harvest rollover and history

Keep original photos, per-ear details, and active harvest summaries while a harvest is active. Starting a new harvest prompts before cleanup and states that photos and individual ear details will be removed while a summary stays on the device.

On confirmation:

1. Build and save the prior harvest summary.
2. Confirm the summary is durable before removing any per-ear data or image files.
3. Delete the prior harvest's original and derived images and per-ear records.
4. Start the new active harvest.

If summary creation fails, retain the old harvest and do not delete its photos or rows. If the user cancels, leave the active harvest unchanged.

The archived summary contains a date label, total ears, and count by grade. Archived harvest history displays these summaries only. It does not retain per-ear rows, images, detections, or detailed results.

## Android and inference architecture

- **UI:** Kotlin and Jetpack Compose.
- **Camera and gallery:** CameraX and Android Photo Picker.
- **Model runtime:** ONNX Runtime Mobile, running on-device.
- **Active and archived harvest data:** Room database.
- **Images:** app-private storage, retained only for the active harvest and removed during confirmed rollover.
- **Network:** inspection and history make no server calls. No account, cloud sync, or network connection is required at runtime.

The ONNX files are exports of these same trained models, not replacement algorithms:

| Task | Model | Source checkpoint |
|---|---|---|
| Find and segment ears in one image | YOLO11s-seg | `data/weights/best.pt` |
| Classify ear variety | EfficientNetV2-S | `data/weights/efficientnetv2_s_corn.pt` |
| Detect defects | YOLO11s | `data/weights/yolo11s_colab_best_2026-10-08.pt` |

Kotlin performs image decoding and EXIF orientation, model input preparation, ear-mask cropping, detection postprocessing, view merging, trait calculation, grading, and utilization recommendations. Preserve the trained preprocessing sizes, color order, normalization, class order, confidence behavior, merge rules, and grade configuration from the existing project. Optimize or quantize only after measuring the exported models and checking that outputs remain acceptable.

Deliver model files with the app installation; do not make a first-run network download a requirement. Use install-time asset delivery for Play distribution if that keeps the base app package practical. The source PyTorch checkpoints found in the related project worktree total about 116 MB before conversion. Final ONNX and installed download sizes are not yet known.

The FastAPI backend, web Dataset page, annotation desk, training code, and evaluation tools stay in the repository as developer workflows. The Android field app does not embed Python or wrap the existing web UI in a WebView.

## Existing project context and gaps

- The current repository has no Android project or ONNX files.
- The active checkout does not contain the trained model weights. Source checkpoints are present in a related Maisagip worktree and are not tracked in Git.
- The README describes the YOLO11s-seg multi-ear flow, but the backend pipeline in this checkout does not yet implement that segmenter. The Android implementation must follow the approved segmentation behavior and the trained checkpoint, rather than assume the current `/inspect` endpoint already provides it.
- The source model checkpoints total approximately 116 MB on disk. Export size, runtime compatibility, device speed, memory use, and any quantization effect need to be measured before setting a final installation-size claim.

## Out of scope

- Cloud accounts, server-side inspection, synchronization, and online history.
- Dataset labeling, training, or model management inside the Android app.
- Treating multiple separate-ear images as separate ears in one inspection. Multiple selected images mean views of one ear; separate ears in a batch are supported when they appear together in one image.
- Food-safety certification or a claim that a visual recommendation establishes safety.
- Replacing or retraining the classifier, detector, segmenter, or current rule system.

## Design self-review

- The data lifecycle preserves active harvest details and photos, saves the summary before deletion, confirms rollover, and keeps only the requested aggregate fields afterward.
- Automatic routing is explicit and consistent with the existing inspection protocol: one image is segmented into ears; multiple images are views of one ear.
- The three named model checkpoints match the user's confirmed YOLO11s-seg, YOLO11s, and EfficientNetV2-S set.
- The runtime and model size remain a feasibility risk; the design does not claim that conversion or quantization is complete or lossless.
