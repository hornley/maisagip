# HEIC Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Accept HEIC/HEIF images in Maisagip, preserve originals, and normalize new and migrated dataset images into oriented RGB JPEGs for reliable training and inference.

**Architecture:** Add one shared \`image_io\` module that registers \`pillow-heif\`, decodes bytes with EXIF orientation correction, and encodes canonical JPEGs. Inference uses its decoder in memory; dataset uploads and a migration command use its normalizer to write canonical JPGs and archive originals under \`data/raw/originals/\` while leaving labels keyed by the unchanged ear/view stem.

**Tech Stack:** Python, Pillow, pillow-heif, NumPy, FastAPI, pytest, vanilla HTML/JavaScript.

---

### Task 1: Shared image format boundary

**Files:**

- Create: \`backend/app/image_io.py\`
- Modify: \`requirements.txt\`
- Create: \`backend/tests/test_image_io.py\`

- [ ] **Step 1: Write failing unit tests for RGB decoding and canonical JPEG encoding**

Add tests that create a small PNG in memory, call \`decode_image_bytes\`, assert a 3-channel RGB NumPy array, call \`encode_canonical_jpeg\`, reopen the returned bytes with Pillow, and assert \`format == "JPEG"\` and \`mode == "RGB"\`.

- [ ] **Step 2: Run the focused tests to verify the new module is absent**

Run:

~~~bash
.venv/bin/python -m pytest backend/tests/test_image_io.py -v
~~~

Expected: collection fails because \`backend.app.image_io\` does not exist.

- [ ] **Step 3: Add the shared decoder and encoder**

Implement \`backend/app/image_io.py\` with these public interfaces:

~~~python
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

def decode_image_bytes(data: bytes) -> np.ndarray: ...
def normalize_image_bytes(data: bytes) -> tuple[np.ndarray, bytes]: ...
def encode_canonical_jpeg(image: Image.Image | np.ndarray) -> bytes: ...
~~~

Register the HEIF opener once at module import using \`pillow_heif.register_heif_opener()\`. Use \`ImageOps.exif_transpose\`, convert to RGB, and return \`np.asarray\`. Raise a \`ValueError\` with the original exception chained when decoding fails. Encode JPEG with quality 95, optimize enabled, and no EXIF metadata.

Add \`pillow-heif>=0.18\` to \`requirements.txt\`.

- [ ] **Step 4: Add codec-gated HEIC tests**

Create a 2×1 RGB image, set an EXIF orientation tag that requires a 90-degree transpose, save it as HEIC through \`pillow_heif.encode\`, and verify \`decode_image_bytes\` returns the transposed dimensions. Mark only this fixture test with \`pytest.importorskip("pillow_heif")\`; the normal requirements installation must run it.

- [ ] **Step 5: Run the image helper tests**

Run:

~~~bash
.venv/bin/python -m pytest backend/tests/test_image_io.py -v
~~~

Expected: all tests pass, or only the HEIC fixture test is skipped when the codec is not installed.

- [ ] **Step 6: Commit the shared image boundary**

~~~bash
git add backend/app/image_io.py backend/tests/test_image_io.py requirements.txt
git commit -m "feat: add shared HEIC image normalization"
~~~

If the workspace cannot write \`.git\`, retain the changes and report that the commit was blocked by filesystem permissions.

### Task 2: HEIC inference support

**Files:**

- Modify: \`backend/app/pipeline.py:1-30\`
- Modify: \`backend/tests/test_pipeline.py\`

- [ ] **Step 1: Write a regression test using the shared decoder**

Patch \`backend.app.pipeline.models.classify_variety\`, \`detect_defects\`, and \`build_report\` as the existing pipeline tests do. Send a PNG through \`inspect_image_bytes\`, then assert the provider receives an RGB array and that \`pipeline.decode_image\` delegates to \`image_io.decode_image_bytes\`.

- [ ] **Step 2: Run the regression test before changing the pipeline**

Run:

~~~bash
.venv/bin/python -m pytest backend/tests/test_pipeline.py -v
~~~

Expected: the new delegation assertion fails because \`pipeline.py\` still opens images directly with Pillow.

- [ ] **Step 3: Replace the local decoder with the shared decoder**

Remove the direct \`io\`/\`PIL.Image\` decoder imports from \`pipeline.py\`, import \`image_io\`, and make \`decode_image(bytes_data)\` return \`image_io.decode_image_bytes(bytes_data)\`. Keep the existing \`ValueError\` API and all inspection behavior unchanged.

- [ ] **Step 4: Run pipeline and full backend tests**

Run:

~~~bash
.venv/bin/python -m pytest backend/tests/test_pipeline.py backend/tests/test_views.py backend/tests/test_rules.py -v
~~~

Expected: all tests pass.

- [ ] **Step 5: Commit inference integration**

~~~bash
git add backend/app/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: decode HEIC images during inspection"
~~~

### Task 3: Canonical dataset uploads and archival originals

**Files:**

- Modify: \`backend/app/config.py\`
- Modify: \`backend/app/dataset_api.py\`
- Modify: \`backend/tests/test_dataset_api.py\`

- [ ] **Step 1: Write failing dataset tests**

Add tests that monkeypatch the test \`DATA_DIR\`, upload a generated HEIC image with \`filename="photo.heic"\`, and assert:

~~~text
data/raw/originals/ear001/photo.heic exists
data/raw/classifier/yellow_sweet_corn/ear001_v1.jpg exists
data/raw/detector/images/ear001_v1.jpg exists
~~~

Reopen the canonical image and assert it is RGB JPEG. Add a test uploading the same visual image once as PNG and once as HEIC and assert the second request returns 400 with a duplicate message. Add a test that a bad label leaves neither archive nor canonical files behind.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

~~~bash
.venv/bin/python -m pytest backend/tests/test_dataset_api.py -v
~~~

Expected: HEIC uploads fail as unsupported and no archive is created.

- [ ] **Step 3: Add archive configuration and image normalization to the dataset API**

Add \`ORIGINALS_DIR = DATA_DIR / "raw" / "originals"\` to \`config.py\`. Import \`image_io\` and set \`IMAGE_EXTS = image_io.SUPPORTED_IMAGE_EXTS\` in \`dataset_api.py\`.

Implement helpers with these behaviors:

~~~python
def _originals_dir() -> Path:
    return config.ORIGINALS_DIR

def _safe_original_name(filename: str) -> str: ...

def _archive_path(ear_id_value: str, filename: str) -> Path: ...
~~~

\`_archive_path\` must prevent path traversal, preserve the original extension, and add \`-2\`, \`-3\`, etc. before the extension if a same-name file already exists.

During upload planning, call \`image_io.normalize_image_bytes\`, hash the normalized bytes (not the incoming bytes), and retain the original bytes plus canonical JPEG bytes in each plan. Run all collision and label validation before any writes. Store the archive copy and canonical JPG copies only after validation succeeds. Always use \`<ear>_v<view>.jpg\` for both dataset branches.

Update duplicate scanning to hash existing canonical file bytes. Keep existing JPG/PNG uploads compatible while ensuring newly stored files are canonical JPGs.

Update deletion to remove archived originals for the ear in addition to canonical images and labels. Update \`_media_type\` to recognize \`.heic\`/\`.heif\` for any legacy files still present.

- [ ] **Step 4: Run dataset tests and the full test suite**

Run:

~~~bash
.venv/bin/python -m pytest backend/tests/test_dataset_api.py -v
.venv/bin/python -m pytest backend/tests -v
~~~

Expected: all tests pass.

- [ ] **Step 5: Commit dataset normalization**

~~~bash
git add backend/app/config.py backend/app/dataset_api.py backend/tests/test_dataset_api.py
git commit -m "feat: normalize and archive HEIC dataset uploads"
~~~

### Task 4: Migrate the existing HEIC sample dataset

**Files:**

- Create: \`training/normalize_heic_dataset.py\`
- Create: \`backend/tests/test_normalize_heic_dataset.py\`
- Modify: \`README.md\`

- [ ] **Step 1: Write migration tests**

Create a temporary dataset tree containing HEIC files under \`classifier/white_corn/\` and \`detector/images/\`, plus a matching detector label. Run the migration function and assert that each HEIC has a canonical JPG beside it, originals are archived under a separate \`originals/\` tree, the label remains named after the unchanged stem, and a corrupt HEIC is reported without stopping other files from converting.

- [ ] **Step 2: Run migration tests to verify they fail**

Run:

~~~bash
.venv/bin/python -m pytest backend/tests/test_normalize_heic_dataset.py -v
~~~

Expected: collection or test failure because the migration module does not exist.

- [ ] **Step 3: Implement the migration command**

Implement:

~~~bash
python -m training.normalize_heic_dataset \
  --input-root data/raw \
  --originals-root data/raw/originals
~~~

The command must scan only \`classifier/*\` and \`detector/images\`, convert \`.heic\`/\`.heif\` files into same-stem \`.jpg\` files, copy each original into the archive preserving its relative dataset path, skip an already-existing canonical JPG only when it decodes successfully, and print converted/skipped/failed counts. It must never overwrite an existing JPG or delete an HEIC automatically. The existing validator and split scripts will ignore the retained HEIC files and consume the new JPGs.

- [ ] **Step 4: Document the migration workflow**

Update \`README.md\` to list HEIC/HEIF as accepted inputs, explain that dataset storage is canonical RGB JPEG plus archived original, add \`pillow-heif\` to setup requirements via \`pip install -r requirements.txt\`, and include the exact migration command above. State that corrupt files are reported and must be reviewed before training.

- [ ] **Step 5: Run migration, validation, and full tests**

Run:

~~~bash
.venv/bin/python -m training.normalize_heic_dataset --input-root data/raw --originals-root data/raw/originals
.venv/bin/python -m training.validate_dataset
.venv/bin/python -m pytest backend/tests -v
~~~

Expected: migration reports its counts, validation consumes canonical JPGs, and all tests pass. Do not run the migration against the real sample tree until the generated file list has been reviewed if the sample files are present.

- [ ] **Step 6: Commit migration and documentation**

~~~bash
git add training/normalize_heic_dataset.py backend/tests/test_normalize_heic_dataset.py README.md
git commit -m "feat: add HEIC dataset migration command"
~~~

### Task 5: Advertise supported formats in the UI

**Files:**

- Modify: \`frontend/index.html:31-34\`
- Modify: \`frontend/dataset.html:63-125\`
- Modify: \`frontend/dataset.js\`
- Modify: \`frontend/styles.css\` only if the format copy requires it

- [ ] **Step 1: Update file input accept hints and visible copy**

Use \`accept=".heic,.heif,.jpg,.jpeg,.png,image/*"\` for inspection and dataset image controls. Change visible hints to \`HEIC · JPG · PNG\` and add a short note that HEIC images are normalized by the server for processing.

- [ ] **Step 2: Run the existing backend suite and inspect the HTML references**

Run:

~~~bash
.venv/bin/python -m pytest backend/tests -v
rg -n "HEIC|HEIF|accept=.*heic" frontend README.md
~~~

Expected: all tests pass and every image upload control advertises the new formats.

- [ ] **Step 3: Commit UI copy changes**

~~~bash
git add frontend/index.html frontend/dataset.html frontend/dataset.js frontend/styles.css
git commit -m "feat: advertise HEIC uploads in the UI"
~~~

## Final verification

After all tasks, run:

~~~bash
.venv/bin/python -m pytest backend/tests -v
.venv/bin/python -m training.validate_dataset
~~~

Start the app with:

~~~bash
.venv/bin/uvicorn backend.app.main:app --reload --port 8000
~~~

Verify \`GET /health\`, upload one HEIC through \`/inspect\`, and import one HEIC through \`/dataset/ears\`. Confirm the report works, the archive contains the original, and both canonical dataset branches contain the JPG.
