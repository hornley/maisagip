import numpy as np
import pytest

from backend.app import report


@pytest.mark.parametrize(
    ("grade", "expected_labels"),
    [
        ("ExtraClass", {"Local Market", "Export", "Premium Sale", "Processed/Canned"}),
        ("ClassI", {"Local Market", "Premium Sale", "Processed/Canned"}),
        ("ClassII", {"Local Market"}),
        ("Reject", {"Poultry Feed"}),
    ],
)
def test_build_report_maps_canonical_grade_to_recommendation_outlets(
    monkeypatch, grade, expected_labels
):
    monkeypatch.setattr(report, "_save", lambda *args: None)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    variety = {"class": "yellow_sweet_corn", "confidence": 0.99}
    merged = {
        "variety": variety,
        "defects": [],
        "defect_coverage": 0.0,
        "severe_present": False,
        "traits": {"ear_size": "large", "kernel_completeness": 1.0},
        "view_count": 1,
        "variety_disagreement": {"present": False},
    }
    engine_result = {"grade": grade, "grade_label": grade.replace("Class", "Class ")}

    result = report.build_report(
        per_view=[
            {
                "image": image,
                "detections": [],
                "variety": variety,
                "traits": merged["traits"],
            }
        ],
        merged=merged,
        engine_result=engine_result,
    )

    assert {option["label"] for option in result["recommendations"]["options"]} == expected_labels
    assert result["grade"]["grade_label"] == engine_result["grade_label"]
