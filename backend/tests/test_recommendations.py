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


def test_accepted_returns_visual_market_candidates_and_current_price():
    result = _result(price_data=[_price()])
    labels = {option["label"] for option in result["options"]}
    assert {"Local market", "Export", "Premium sale", "Food processing/canned"} <= labels
    local = next(o for o in result["options"] if o["label"] == "Local market")
    assert local["status"] == "candidate"
    assert local["price_estimate"]["low"] == 10
    assert "required_external_checks" in local


def test_reject_has_safe_practical_alternative():
    result = _result("Reject")
    feed = next(o for o in result["options"] if o["label"] == "Poultry/feed alternative")
    assert feed["status"] == "requires_check"


def test_mold_omits_feed_and_advises_hold_testing():
    result = _result("Reject", ["mold"])
    labels = {option["label"] for option in result["options"]}
    assert "Poultry/feed alternative" not in labels
    hold = next(o for o in result["options"] if o["label"] == "Hold for qualified inspection/testing")
    assert hold["status"] == "requires_check"


def test_stale_price_is_omitted():
    result = _result(price_data=[_price(valid_through="2026-10-08")])
    local = next(o for o in result["options"] if o["label"] == "Local market")
    assert "price_estimate" not in local


def test_missing_adjustment_factor_is_omitted():
    result = _result(price_data=[_price(adjustment_factors={"defect_coverage": 1, "size": 1})])
    local = next(o for o in result["options"] if o["label"] == "Local market")
    assert "price_estimate" not in local
