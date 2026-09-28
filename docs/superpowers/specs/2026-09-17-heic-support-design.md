# HEIC Image Support Design

## Goal

Allow Maisagip to accept HEIC and HEIF images while preserving the original uploads and using canonical, oriented RGB JPEGs for inference, annotation, validation, splitting, and training.

## Context

The inspection pipeline decodes uploaded bytes through Pillow. Pillow does not decode HEIC/HEIF unless a plugin is installed. The dataset API and training utilities currently recognize only `.jpg`, `.jpeg`, and `.png`, and the browser cannot reliably preview HEIC files across all platforms.

The current sample dataset contains HEIC images. Existing uncommitted dataset-uploader changes must be preserved; this feature should add only the image-format boundary and its related tests/docs.

## Chosen approach

Use `pillow-heif` as the HEIC/HEIF decoder and normalize images at the application boundary:

1. Accept `.heic` and `.heif` in uploads.
2. Decode the image and apply EXIF orientation before any image is used or saved.
3. Convert to 8-bit RGB.
4. Save canonical copies as `.jpg` using quality 95.
5. Preserve the original upload under `data/raw/originals/<ear_id>/` for dataset provenance.

JPG and PNG uploads will use the same normalization path for new dataset imports. Existing JPG/PNG files will not be rewritten automatically.

## Architecture

Create `backend/app/image_io.py` as the shared image-format boundary. It owns:

- HEIF opener registration.
- Supported input suffixes and MIME-independent format recognition.
- Byte decoding with EXIF orientation correction.
- RGB conversion.
- Canonical JPEG encoding.

`backend/app/pipeline.py` will call the shared decoder for inference and return the same RGB NumPy array as before. `backend/app/dataset_api.py` will call the shared decoder for validation and use the canonical encoder for stored training images. This prevents the two entry points from developing different HEIC behavior.

The dataset API will store originals separately from the canonical dataset tree:

```text
data/raw/originals/<ear_id>/<original filename>
data/raw/classifier/<variety>/<ear_id>_v<view>.jpg
data/raw/detector/images/<ear_id>_v<view>.jpg
data/raw/detector/labels/<ear_id>_v<view>.txt
```

The classifier and detector copies will contain identical canonical JPEG bytes. Labels will continue to be named by canonical image stem and will use the displayed, orientation-corrected image dimensions.

## Dataset upload behavior

For each image upload, the API will:

- Read the upload bytes.
- Reject empty or unsupported files with the existing HTTP 400 pattern.
- Decode and normalize the image.
- Detect duplicates using the canonical RGB/JPEG representation, so the same image uploaded as HEIC, PNG, and JPG is treated as a duplicate.
- Check all requested labels before writing files.
- Write the original upload to the archive and the canonical JPEG to both dataset branches only after validation succeeds.

Original archive names will be made safe for use as path components. If two uploads for the same ear have the same basename, the API will preserve both without overwriting by adding a deterministic suffix.

Deleting an ear will remove its canonical images, labels, and archived originals so the archive does not retain orphaned personal images.

## Inference behavior

The `/inspect` endpoint will accept HEIC/HEIF bytes without relying on the browser-provided content type or filename extension. The decoder will apply EXIF orientation and convert to RGB before demo or real model providers receive the image.

The response and generated report images remain PNG as they are today.

## Frontend behavior

All image file controls will advertise `.heic`, `.heif`, `.jpg`, `.jpeg`, and `.png`. The interface copy will describe the accepted formats. Client-side previews may remain unavailable in browsers without HEIC support; the server's decode error will be shown if the image cannot be processed.

## Error handling and compatibility

- A missing or unusable HEIF decoder will be surfaced as a clear installation/runtime error rather than a generic invalid-image message.
- Corrupt or unsupported image data will remain a client error.
- Training and evaluation scripts will continue to enumerate canonical JPG/PNG files only.
- No model changes, label-coordinate transformations, or changes to the grading rules are required.

## Testing

Add unit tests for the shared image helper using existing PNG fixtures and a small generated HEIC fixture when `pillow-heif` is available. Tests will verify:

- PNG/JPG decoding returns an RGB NumPy array.
- EXIF orientation is applied before conversion.
- Canonical encoding produces decodable RGB JPEG bytes.
- HEIC/HEIF suffixes are accepted by dataset upload validation.
- Dataset upload stores an original and canonical copies, with labels still matching the canonical stem.
- Canonical duplicate detection catches equivalent images across input formats.
- `/inspect` uses the shared decoder for HEIC bytes.

Codec-dependent tests will skip with an explicit reason when `pillow-heif` is unavailable, while the dependency will be included in the normal requirements so a standard installation runs them.

## Documentation

Update `README.md` with the HEIC support requirement, canonical-storage policy, and a one-time command for importing the current HEIC sample dataset through the dataset uploader or migration script. The migration path must preserve originals and must not silently discard files that fail to decode.
