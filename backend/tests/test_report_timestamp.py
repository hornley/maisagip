from datetime import datetime
import re

import numpy as np

from app import report


def test_built_report_inspection_timestamp_is_timezone_aware(monkeypatch, tmp_path):
    monkeypatch.setattr(report.config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(report, "_save", lambda *args: tmp_path / "report.png")

    per_view = [{
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "detections": [],
        "variety": {"class": "white_corn"},
        "traits": {},
    }]
    merged = {
        "variety": {"class": "white_corn"},
        "defects": [],
        "defect_coverage": 0.0,
        "severe_present": False,
        "traits": {},
        "view_count": 1,
        "variety_disagreement": {"present": False},
    }
    engine_result = {"grade_label": "ExtraClass"}

    built = report.build_report(per_view, merged, engine_result)
    parsed = datetime.fromisoformat(built["inspection_date"])

    assert re.search(r"[+-]\d{2}:\d{2}$", built["inspection_date"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
