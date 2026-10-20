# Demo presentation and history clearing

## Scope

Add a `python -m backend.app.main --demo` launch mode that trims the frontend for presentations. It hides the Dataset navigation link, the Ear appearance and Kernel uniformity pilot cards, ear size, and merged-defect counts. The flag changes presentation only; model provider selection stays governed by the existing weight-loading behavior.

Add a Clear history control to the History page. After browser confirmation, it deletes saved report records, cached reports, and generated report images. The page then shows its empty-history state. Failed requests keep the current history visible and show an error.

## Implementation shape

- Keep the existing FastAPI `app` importable by Uvicorn and add a small module CLI for `--demo`.
- Expose the presentation flag through a small UI configuration endpoint consumed by the inspection and history pages.
- Add `DELETE /history` and a report-store helper that removes report rows and their generated images.
- Keep defect counts visible in the full interface; omit them only in `--demo` mode.

## Error handling and data impact

The browser asks for confirmation before clearing. The server clears the in-memory cache and SQLite records along with the report image files. A failed HTTP request is shown on the History page without clearing its visible list.

## Verification

Review the changed source paths and diff. No test files or test runs are included in this change.
