from datetime import date, timedelta

from backend.app.recommendations import build_recommendations


def _price(**overrides):
    row = {
        "market": "local_market", "variety": "yellow_sweet_corn", "grade_product_basis": "ClassI",
        "currency": "PHP", "unit": "kg", "low": 10, "high": 12,
        "adjustment_factors": {"defect_coverage": 1, "size": 1, "completeness": 1},
        "source": "cooperative bulletin", "effective_date": "2026-10-01", "valid_through": "2026-10-31",
    }
    row.update(overrides)
    return row


def _result(grade="ClassI", defects=None, price_data=None, **traits):
    return build_recommendations(grade, "yellow_sweet_corn", defects or [], traits or {"ear_size": "large", "kernel_completeness": .96}, price_data, "2026-10-09")


def _labels(grade, defects=None, price_data=None):
    return {option["label"] for option in _result(grade, defects, price_data)["options"]}


def test_grade_filters_outlet_labels_exactly():
    assert _labels("ExtraClass") == {"Local Market", "Export", "Premium Sale", "Processed/Canned"}
    assert _labels("ClassI") == {"Local Market", "Premium Sale", "Processed/Canned"}
    assert _labels("ClassII") == {"Local Market", "Processed/Canned"}
    assert _labels("Reject") == set()


def test_mold_leaves_only_local_market_for_non_reject_grades():
    for grade in ("ExtraClass", "ClassI", "ClassII"):
        assert _labels(grade, ["mold"]) == {"Local Market"}
    assert _labels("Reject", ["mold"]) == set()


def test_ear_decay_leaves_only_local_market_for_non_reject_grades():
    for grade in ("ExtraClass", "ClassI", "ClassII"):
        assert _labels(grade, ["ear_decay"]) == {"Local Market"}
    assert _labels("Reject", ["ear_decay"]) == set()


def test_unrelated_defects_do_not_add_grade_excluded_outlets():
    assert _labels("ClassI", ["insect_damage"]) == {"Local Market", "Premium Sale", "Processed/Canned"}
    assert _labels("ClassII", ["insect_damage"]) == {"Local Market", "Processed/Canned"}
    assert _labels("Reject", ["insect_damage"]) == set()


def test_options_only_expose_label_and_validated_price():
    result = _result(price_data=[_price()])
    local = next(o for o in result["options"] if o["label"] == "Local Market")
    assert local["price_estimate"]["low"] == 10
    assert set(local) == {"label", "price_estimate"}
    for option in result["options"]:
        assert not {"status", "reason", "required_external_checks"}.intersection(option)


def test_price_is_never_emitted_for_ineligible_channel():
    result = _result("ClassI", price_data=[_price(market="export")])
    assert all("price_estimate" not in option for option in result["options"])


def test_stale_price_is_omitted_from_eligible_local_market_option():
    result = _result(price_data=[_price(valid_through="2026-10-08")])
    local = next(o for o in result["options"] if o["label"] == "Local Market")
    assert set(local) == {"label"}


def test_missing_adjustment_factor_is_omitted_from_eligible_local_market_option():
    result = _result(price_data=[_price(adjustment_factors={"defect_coverage": 1, "size": 1})])
    local = next(o for o in result["options"] if o["label"] == "Local Market")
    assert set(local) == {"label"}
