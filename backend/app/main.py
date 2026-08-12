from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, models
from .pipeline import inspect_ear
from .report import REPORTS

app = FastAPI(title="Maisagip", version="0.2.0")


@app.post("/inspect")
async def inspect(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    contents = []
    for file in files:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Empty upload: {file.filename}")
        contents.append(content)
    try:
        return inspect_ear(contents)
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