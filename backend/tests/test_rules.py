from backend.app.rules.engine import GradeEngine

engine = GradeEngine()


def grade(coverage, severe=False, vconf=0.9, confs=()):
    return engine.evaluate(vconf, coverage, severe, 1.0, list(confs))


def test_zero_coverage_is_extra_class_human_consumption():
    result = grade(0.0)
    assert result["grade"] == "ExtraClass"
    assert result["grade_label"] == "Extra Class"
    assert result["utilization"]["recommendation"] == "human_consumption"
    assert result["grade_reasons"]


def test_under_five_percent_is_class_one():
    result = grade(0.03)
    assert result["grade"] == "ClassI"
    assert result["utilization"]["recommendation"] == "human_consumption"


def test_between_five_and_ten_percent_is_class_two():
    result = grade(0.07)
    assert result["grade"] == "ClassII"
    assert result["utilization"]["recommendation"] == "food_processing"


def test_coverage_at_ten_percent_is_rejected():
    result = grade(0.10)
    assert result["grade"] == "Reject"
    assert result["grade_label"] == "Reject"
    assert result["utilization"]["recommendation"] == "reject"


def test_coverage_above_ten_percent_is_rejected():
    result = grade(0.14)
    assert result["grade"] == "Reject"
    assert result["grade_label"] == "Reject"
    assert result["utilization"]["recommendation"] == "reject"


def test_five_percent_boundary_goes_to_class_two():
    result = grade(0.05)
    assert result["grade"] == "ClassII"


def test_severe_defect_forces_animal_feed_even_at_low_coverage():
    result = grade(0.02, severe=True)
    assert result["utilization"]["recommendation"] == "animal_feed"
    assert result["severe_present"] is True


def test_severe_defect_with_high_coverage_still_rejected():
    result = grade(0.12, severe=True)
    assert result["utilization"]["recommendation"] == "reject"


def test_coverage_is_reported_verbatim():
    result = grade(0.068)
    assert result["defect_coverage"] == 0.068


def test_low_confidence_triggers_reinspection():
    result = grade(0.02, vconf=0.4)
    assert result["needs_reinspection"] is True


def test_low_defect_confidence_triggers_reinspection():
    result = grade(0.02, confs=[0.2])
    assert result["needs_reinspection"] is True


def test_high_confidence_no_reinspection():
    result = grade(0.02, vconf=0.95, confs=[0.9])
    assert result["needs_reinspection"] is False


def test_grade_reasons_are_explainable():
    result = grade(0.0)
    summary = "\n".join(result["grade_reasons"])
    assert "Meets Extra Class criteria" in summary


def test_grade_reasons_use_percentages():
    result = grade(0.2643)
    summary = "\n".join(result["grade_reasons"])
    assert "26.43%" in summary
    assert "5%" in summary
    assert "10%" in summary
    assert "0.2643" not in summary
