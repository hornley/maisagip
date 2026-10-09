from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, models
from .dataset_api import router as dataset_router
from .pipeline import InconsistentInspectionInput, inspect_ear
from .report import REPORTS

app = FastAPI(title="Maisagip", version="0.2.0")
app.include_router(dataset_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    import traceback

    traceback.print_exception(type(exc), exc, exc.__traceback__)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


@app.post("/inspect")
async def inspect(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(
        config.DEFAULT_CONFIDENCE_THRESHOLD,
        ge=config.MIN_CONFIDENCE_THRESHOLD,
        le=config.MAX_CONFIDENCE_THRESHOLD,
    ),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    contents = []
    for file in files:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Empty upload: {file.filename}")
        contents.append(content)
    try:
        return inspect_ear(contents, confidence_threshold)
    except InconsistentInspectionInput as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inspection failed: {exc}") from exc


@app.get("/report/{report_id}")
def get_report(report_id: str):
    report = REPORTS.get(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/report/{report_id}/views/grid/image")
def get_report_grid_image(report_id: str):
    return _report_image(report_id, "grid")


@app.get("/report/{report_id}/views/{view_index}/image")
def get_report_view_image(report_id: str, view_index: int):
    return _report_image(report_id, f"v{view_index}")


def _report_image(report_id: str, suffix: str):
    path = config.REPORTS_DIR / f"{report_id}_{suffix}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report image not found.")
    return FileResponse(str(path), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok", "providers": models.provider_modes()}


config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
