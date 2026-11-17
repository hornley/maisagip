# Maisagip Android App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native Android app that inspects corn offline with the approved three ONNX models and keeps per-ear details only for the active harvest.

**Architecture:** Add a Kotlin/Jetpack Compose app under `android/`; keep the existing Python backend and dataset workflows intact. Export the existing PyTorch checkpoints to fixed-shape ONNX assets, run segmentation/classification/detection with ONNX Runtime Mobile, and port the current merge and rule behavior to Kotlin. Store active harvests in Room and private image files; archive only an aggregate summary after a confirmed rollover.

**Tech Stack:** Kotlin, Jetpack Compose, CameraX, Android Photo Picker, Room, ONNX Runtime Mobile, Python/PyTorch/Ultralytics for model export.

---

## Scope and implementation constraints

- The current repository has no Android module or ONNX exports. Source checkpoints are in `../t3-44db0360/data/weights/` and total about 116 MB before conversion; do not commit checkpoint files.
- Export model files locally into Android assets. Measure each ONNX file and the resulting APK/AAB before choosing base-module packaging or an install-time asset pack. Installation must include all three models; inference must not download them.
- Preserve the current Python behavior: RGB/EXIF image handling, classifier 448-pixel resize/crop and ImageNet normalization, YOLO input size 640, 0.50 default confidence, detector NMS IoU 0.7, view merging, trait calculations, `backend/app/rules/config_rules.json` grading/utilization thresholds, and `backend/app/recommendations.py` outlet eligibility. The detector checkpoint outputs eight classes in its trained order (`corn_ear`, `mold`, `insect_damage`, `discoloration`, `deformity`, `missing_kernels`, `ear_decay`, `shriveled_kernels`); canonicalize both `mold` and `ear_decay` to `decay`, matching `backend/app/config.py`. The segmenter uses class 0 (`corn-ear`, canonical ear label `corn_ear`) and confidence 0.25 from the existing multi-ear integration.
- The approved history scope is a date, ear total, and counts by grade only. Do not add the daily/weekly and per-defect breakdowns from related issue #2.
- Never fall back to a demo model. If any required model is missing, malformed, or incompatible, show a clear blocking error.
- Do not delete the current Python tools or edit unrelated dirty files. Stage only Android app files, the export tool, and this plan when committing.

## File map

- `android/settings.gradle.kts`, `android/build.gradle.kts`, `android/gradle.properties`, `android/app/build.gradle.kts`, `android/app/src/main/AndroidManifest.xml` — standalone Android build, app dependencies, permissions, and model asset packaging.
- `android/app/src/main/java/.../MainActivity.kt` — Compose entry point and navigation host.
- `android/app/src/main/java/.../data/HarvestDatabase.kt`, `HarvestDao.kt`, `HarvestRepository.kt` — Room schema, active/archived harvest operations, and rollover transaction.
- `android/app/src/main/java/.../inference/OnnxModels.kt`, `ImagePreprocessor.kt`, `YoloPostprocessor.kt`, `InspectionPipeline.kt`, `ViewMerger.kt`, `GradeRules.kt`, `RecommendationRules.kt` — model sessions, tensor conversion, output decoding, pipeline routing, view merge, traits, grading, and outlet recommendations.
- `android/app/src/main/java/.../ui/HarvestHomeScreen.kt`, `MediaSelectionScreen.kt`, `InspectionScreen.kt`, `EarResultScreen.kt`, `HarvestHistoryScreen.kt`, `SettingsScreen.kt` — minimalist harvest, media, result, history, and advanced threshold screens.
- `android/app/src/main/assets/inspection_rules.json`, `recommendation_rules.json` — checked-in rules copied from the current backend configurations.
- `android/app/src/main/assets/models/model_manifest.json` — checked-in export metadata describing the locally generated ONNX model assets.
- `training/export_android_models.py` — reproducible export of the three named checkpoints to app asset paths; generated ONNX files remain local build inputs.
- `.gitignore` — exclude local model exports and Android build output, without excluding the export script or rule source.
- `README.md` — Android build, model export, and install instructions; keep existing Python quickstart and training documentation.

## Task 1: Create the Android project shell

**Files:**
- Create: `android/settings.gradle.kts`
- Create: `android/build.gradle.kts`
- Create: `android/gradle.properties`
- Create: `android/app/build.gradle.kts`
- Create: `android/app/src/main/AndroidManifest.xml`
- Create: `android/app/src/main/java/com/maisagip/app/MainActivity.kt`

- [ ] **Step 1: Add the app module and wrapper**

Create a single `:app` application module using the current stable Android Gradle Plugin, Kotlin Android plugin, Compose compiler, and Gradle wrapper. Set `minSdk 28` so Android's built-in image decoder supports HEIF/HEIC, and keep Compose and Java/Kotlin targets consistent. Add Compose Material 3, Navigation Compose, Room KTX/compiler, CameraX, and ONNX Runtime Android dependencies only.

- [ ] **Step 2: Add the app entry point and permissions**

Declare one launcher activity and the camera permission. Do not request storage permission: use Android Photo Picker and app-private files. Do not declare `INTERNET`; model inference and harvest history are local. Build a minimal Compose `Scaffold` with navigation destinations for Harvest, History, and Settings.

- [ ] **Step 3: Build the empty app module**

Run `cd android && ./gradlew :app:assembleDebug`.

Expected: a debug APK builds without requiring the Python backend or model checkpoints.

## Task 2: Export and load the trained models

**Files:**
- Create: `training/export_android_models.py`
- Create: `android/app/src/main/java/com/maisagip/app/inference/OnnxModels.kt`
- Create: `android/app/src/main/java/com/maisagip/app/inference/ImagePreprocessor.kt`
- Create: `android/app/src/main/java/com/maisagip/app/inference/YoloPostprocessor.kt`
- Modify: `.gitignore`

- [ ] **Step 1: Define the model export contract**

Export these exact checkpoints: `best.pt` as YOLO11s-seg, `efficientnetv2_s_corn.pt` as EfficientNetV2-S, and `yolo11s_colab_best_2026-10-08.pt` as YOLO11s. Write fixed-shape ONNX assets to `android/app/src/main/assets/models/` and a small manifest beside them with input dimensions, RGB order, normalization, class order, confidence thresholds, IoU threshold, and output tensor names/shapes. Do not change the training checkpoints.

- [ ] **Step 2: Implement the exporter**

Use Ultralytics ONNX export for the two YOLO checkpoints with 640-pixel fixed inputs and embedded NMS disabled so Kotlin can apply the same confidence/NMS rules. Build EfficientNetV2-S with the same two-output classifier head and export it with a 448-pixel input. Use static batch size 1 and an ONNX opset supported by ONNX Runtime Mobile. Fail with a path-specific error if any source checkpoint is missing. Write each model's trained class names/order, input contract, output metadata, and SHA-256 to the tracked `model_manifest.json`; the `.onnx` files remain local build inputs. Do not replace the detector checkpoint's eight-channel output order with the seven-label canonical reporting order.

Command:

```bash
python training/export_android_models.py \
  --weights-dir ../t3-44db0360/data/weights \
  --output-dir android/app/src/main/assets/models
```

- [ ] **Step 3: Exclude local binaries and build output**

Ignore only generated `*.onnx` files under `android/app/src/main/assets/models/` and Android `build/` directories. Keep the manifest, exporter, and configuration tracked. In Gradle, mark ONNX assets uncompressed, copy each asset once into app-private files, and open ONNX Runtime sessions from those file paths. The app build must fail clearly when an asset is absent rather than produce a misleading installable APK.

- [ ] **Step 4: Add fail-closed ONNX session loading**

Open each model from app assets through ONNX Runtime Mobile, verify required inputs/outputs against the manifest, and expose specific loading errors to the UI. Reuse one session per model for the app lifetime and close sessions from the application lifecycle. Do not silently substitute a heuristic.

## Task 3: Port the inference pipeline and rules

**Files:**
- Create: `android/app/src/main/assets/inspection_rules.json`
- Create: `android/app/src/main/java/com/maisagip/app/inference/InspectionPipeline.kt`
- Create: `android/app/src/main/java/com/maisagip/app/inference/ViewMerger.kt`
- Create: `android/app/src/main/java/com/maisagip/app/inference/GradeRules.kt`
- Modify: `android/app/src/main/java/com/maisagip/app/inference/YoloPostprocessor.kt`

- [ ] **Step 1: Implement image preparation and segmentation decoding**

Decode JPEG/PNG/HEIC inputs to upright RGB pixels using Android EXIF metadata support. Reproduce YOLO letterboxing, padding, output-coordinate unletterboxing, confidence filtering, box NMS, and segmentation prototype-mask decoding. For a one-image inspection, crop each mask's bounding rectangle, replace pixels outside the mask with RGB 127, and run classifier and detector inference on each crop. Discard empty or invalid masks and return a recoverable no-ears result when none remain.

- [ ] **Step 2: Implement classifier and detector postprocessing**

Resize the masked ear crop to the EfficientNetV2-S 448-pixel training transform (resize to 1.1×, center crop, RGB float tensor, ImageNet mean/std normalization). Decode the two variety logits in the exact order `white_corn`, `yellow_sweet_corn`. Decode the detector's eight output channels using the checkpoint's trained class order, then map both `mold` and `ear_decay` to `decay`; apply the default 0.50 threshold and 0.7 IoU NMS and retain each detection's canonical class, confidence, and original-image box.

- [ ] **Step 3: Implement automatic routing and multi-view merge**

If the selected input count is one, segment the photo and create one result per usable ear. If it is greater than one, analyze all images as views of one ear and port `backend/app/views.py` behavior: confidence-weighted tie break for variety, confident-variety disagreement flag, adjacent-view defect deduplication by area ratio, summed capped defect coverage, worst-view kernel completeness, and longest-view ear size. Do not add an inspection-type selector.

- [ ] **Step 4: Port trait and grade behavior**

Port `backend/app/traits.py` and `backend/app/rules/engine.py` with the existing constants. Load the same grade/utilization rule data into the app. Also port `backend/app/recommendations.py` eligibility rules and `backend/app/rules/config_recommendations.json`: show the same eligible outlet labels, with no price estimate while the configured rates list is empty. Keep the default confidence at 0.50 and expose the existing 0.50–0.80 adjustment only in Settings. Mark low-confidence or confidently conflicting results as requiring review. Do not infer food or feed safety from images.

## Task 4: Store active harvests and archive safely

**Files:**
- Create: `android/app/src/main/java/com/maisagip/app/data/HarvestDatabase.kt`
- Create: `android/app/src/main/java/com/maisagip/app/data/HarvestDao.kt`
- Create: `android/app/src/main/java/com/maisagip/app/data/HarvestRepository.kt`

- [ ] **Step 1: Define the local records**

Use one Room harvest table with id, start date, active/archive state, total ears, and counts for each grade. Use an ear-result table linked to its harvest with timestamp, grade, variety, serialized detail fields, and source image names. Copy selected gallery URIs into a private staging directory before inspection; CameraX writes captures there. After a successful inference, move staged originals and any detail preview into `files/harvests/{harvestId}/` before inserting result rows. Never write images to shared storage.

- [ ] **Step 2: Save each completed result to the active harvest**

After successful inference, move the staged originals into the active harvest directory, then insert all ear results in one Room transaction. If the database insert fails, keep staged/harvest files available for retry and clean unreferenced files during startup recovery. A one-image multi-ear inspection inserts one row per segmented ear; a multi-image view inspection inserts one merged ear row. Do not delete staged source files when inference fails or the app closes.

- [ ] **Step 3: Implement confirmed rollover with recovery**

On confirmation, calculate and persist the summary and mark the harvest archived in one Room transaction, then commit before deleting any image directory. Remove per-ear rows in that same committed transaction. After commit, recursively delete the prior harvest directory. On app startup, remove directories whose harvest is already archived, covering a crash after the database commit but before file deletion. If the transaction fails, retain the old rows and files and do not create the new active harvest.

- [ ] **Step 4: Keep archived records aggregate-only**

Archived history displays only date, total ears, and counts by grade. It has no relation or route to ear rows, images, detections, or detailed results.

## Task 5: Add capture, inspection, and harvest screens

**Files:**
- Create: `android/app/src/main/java/com/maisagip/app/ui/HarvestHomeScreen.kt`
- Create: `android/app/src/main/java/com/maisagip/app/ui/MediaSelectionScreen.kt`
- Create: `android/app/src/main/java/com/maisagip/app/ui/InspectionScreen.kt`
- Create: `android/app/src/main/java/com/maisagip/app/ui/EarResultScreen.kt`
- Create: `android/app/src/main/java/com/maisagip/app/ui/HarvestHistoryScreen.kt`
- Create: `android/app/src/main/java/com/maisagip/app/ui/SettingsScreen.kt`
- Modify: `android/app/src/main/java/com/maisagip/app/MainActivity.kt`

- [ ] **Step 1: Build the harvest-first home and history**

Show the active harvest, ear rows, grade summary, and one primary action to inspect more corn. If no harvest is active, show one action to start a harvest. Make active ear rows open their detailed result. History shows summary cards only.

- [ ] **Step 2: Add capture and gallery selection**

Use CameraX for **Take a photo** and Android Photo Picker for **Choose from gallery**. Request camera permission only when CameraX is opened. Let the user choose multiple gallery images, show thumbnails, and label the inferred path briefly: one image will be segmented into ears; multiple images will be combined as views of one ear. Provide replacement/removal controls before inspection; do not ask the user to choose an inspection type.

- [ ] **Step 3: Connect inference and result details**

Add a clear **Start inspection** action after media review. Show a concise progress state, then list each detected ear for a single photo or one merged result for multiple photos. Tapping a result opens grade, variety, annotated image, defect coverage, completeness, detected defects, grade rationale, utilization recommendation, eligible outlet labels, and reinspection state. Include the brief visual-safety notice from the approved design. Provide retake/change-photo recovery for empty segmentation, mixed confident varieties, invalid images, or model errors.

- [ ] **Step 4: Add safe harvest rollover and threshold settings**

Starting a new harvest while one is active opens a confirmation that says photos and individual results will be removed while the compact summary remains. Starting the first harvest needs no cleanup confirmation. Cancel leaves the active harvest unchanged. Keep the confidence setting in Settings, constrained to 0.50–0.80, defaulting to 0.50 and persisted with Android `SharedPreferences`.

## Task 6: Package and document the Android build

**Files:**
- Modify: `README.md`
- Modify: `android/app/build.gradle.kts`
- Modify: `.gitignore`

- [ ] **Step 1: Measure exports and package size**

Record each ONNX file size and the built APK/AAB size. Keep models in the base app when the resulting install remains practical; otherwise add one install-time asset pack so all models arrive with installation. In either case, keep first-run inspection independent of network access.

- [ ] **Step 2: Document model export and app build**

Add Android prerequisites, checkpoint locations, exact export command, local model asset requirements, `./gradlew :app:assembleDebug`, APK location, and the existing Python workflow to the README. State measured model/package sizes without claiming accuracy or speed that has not been measured.

- [ ] **Step 3: Complete workflow validation**

Build the debug APK with all three model assets. On an Android device, inspect one photo containing multiple ears and a multi-view ear; verify active details survive an app restart, canceling rollover preserves data, confirming rollover keeps only the summary and deletes image files, and airplane mode does not block inspection. Report any unavailable device/runtime validation explicitly before opening a PR.

## Workflow record

- Issue: https://github.com/hornley/maisagip/issues/4
- Related harvest-history issue: https://github.com/hornley/maisagip/issues/2
- Approved spec: `docs/superpowers/specs/2026-11-17-maisagip-android-app-design.md`
