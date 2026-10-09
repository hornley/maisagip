"""Standalone, conservative recommendations derived from inspection evidence.

This module deliberately does not make safety, certification, moisture, weight, or
market-price claims.  Price rows are supplied by the caller and are treated as
untrusted external data until every required field is validated.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict


CONFIG_PATH = Path(__file__).parent / "rules" / "config_recommendations.json"
PRICE_FIELDS = {
    "market", "variety", "grade_product_basis", "currency", "unit", "low",
    "high", "adjustment_factors", "source", "effective_date", "valid_through",
}


class PriceRecord(TypedDict):
    """External baseline row accepted by the conservative price validator."""

    market: str
    variety: str
    grade_product_basis: str
    currency: str
    unit: str
    low: float
    high: float
    adjustment_factors: Mapping[str, Any]
    source: str
    effective_date: str
    valid_through: str


class PriceEstimate(TypedDict):
    market: str
    currency: str
    unit: str
    low: float
    high: float
    source: str
    effective_date: str
    uncertainty: str


def _config() -> Mapping[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _defect_view(defects: Any) -> tuple[set[str], float | None]:
    """Read only explicit visual defect names and explicit coverage."""
    if isinstance(defects, Mapping):
        raw_classes = defects.get("classes", defects.get("defects", []))
        coverage = defects.get("coverage")
    else:
        raw_classes, coverage = defects or [], None
    if isinstance(raw_classes, str):
        raw_classes = [raw_classes]
    names: set[str] = set()
    for item in raw_classes if isinstance(raw_classes, Sequence) else []:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, Mapping) and isinstance(item.get("class"), str):
            names.add(item["class"])
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        coverage = None
    elif not math.isfinite(float(coverage)) or not 0 <= float(coverage) <= 1:
        coverage = None
    return names, None if coverage is None else float(coverage)


def _grade_key(grade: Any) -> str:
    if isinstance(grade, Mapping):
        grade = grade.get("grade", grade.get("label"))
    return str(grade) if isinstance(grade, str) else ""


def _matching_price(record: Any, channel: str, variety: Any, grade: str,
                    as_of: date, required_adjustments: Sequence[str]) -> dict[str, Any] | None:
    if not isinstance(record, Mapping) or not PRICE_FIELDS.issubset(record):
        return None
    if record.get("market") != channel or record.get("variety") != variety:
        return None
    basis = record.get("grade_product_basis")
    if not isinstance(basis, str) or not (basis == grade or basis in {"all", "any"}):
        return None
    if record.get("currency") != "PHP" or record.get("unit") not in {"kg", "PHP/kg"}:
        return None
    low, high = record.get("low"), record.get("high")
    if (isinstance(low, bool) or isinstance(high, bool) or
            not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or
            not math.isfinite(float(low)) or not math.isfinite(float(high)) or
            float(low) < 0 or float(high) < float(low)):
        return None
    effective, valid_through = _as_date(record.get("effective_date")), _as_date(record.get("valid_through"))
    if effective is None or valid_through is None or effective > as_of or as_of > valid_through:
        return None
    factors = record.get("adjustment_factors")
    if not isinstance(factors, Mapping):
        return None
    multiplier = 1.0
    for name in required_adjustments:
        factor = factors.get(name)
        if isinstance(factor, Mapping):
            factor = factor.get("factor")
        if (isinstance(factor, bool) or not isinstance(factor, (int, float)) or
                not math.isfinite(float(factor)) or float(factor) <= 0):
            return None
        multiplier *= float(factor)
    source = record.get("source")
    if not isinstance(source, str) or not source.strip():
        return None
    return {
        "market": channel,
        "currency": "PHP",
        "unit": "kg",
        "low": round(float(low) * multiplier, 2),
        "high": round(float(high) * multiplier, 2),
        "source": source.strip(),
        "effective_date": effective.isoformat(),
        "uncertainty": "indicative range; confirm with the buyer",
    }


def build_recommendations(
    grade: str,
    variety: str,
    defects: Any,
    traits: Mapping[str, Any] | None,
    price_data: Sequence[PriceRecord] | Mapping[str, Any] | None = None,
    inspection_date: date | str | None = None,
) -> dict[str, Any]:
    """Build visual-evidence-only market and next-action recommendations.

    ``price_data`` may be a list of records or ``{"rates": [...]}``. Invalid,
    incomplete, stale, or mismatched records are ignored rather than repaired.
    """
    policy = _config()
    grade_key = _grade_key(grade)
    variety_text = variety if isinstance(variety, str) else ""
    traits = traits if isinstance(traits, Mapping) else {}
    defect_names, explicit_coverage = _defect_view(defects)
    safety = sorted(defect_names.intersection(policy["safety_exclusions"]))
    as_of = _as_date(inspection_date) or date.today()
    rows = policy.get("rates", []) if price_data is None else (
        price_data.get("rates", []) if isinstance(price_data, Mapping) else price_data
    )
    rows = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []

    coverage_text = (f"explicit visible defect coverage is {explicit_coverage:.1%}"
                     if explicit_coverage is not None else "visible defect coverage was not supplied")
    completeness = traits.get("kernel_completeness")
    if (isinstance(completeness, bool) or not isinstance(completeness, (int, float)) or
            not math.isfinite(float(completeness)) or not 0 <= float(completeness) <= 1):
        completeness = None
    size = traits.get("ear_size")
    options = []
    for channel in policy["channels"]:
        if safety:
            status, reason = "unavailable", (
                "A configured safety exclusion is visible (" + ", ".join(safety) +
                "); hold the lot pending qualified inspection/testing."
            )
        elif grade_key == "Reject":
            status, reason = "unavailable", "The assigned Reject grade does not support this outlet as a current candidate."
        elif channel["id"] == "premium_sale" and (size not in {"large", "medium"} or completeness is not None and completeness < 0.9):
            status, reason = "requires_check", "Visible size/completeness may limit premium acceptance; confirm the buyer specification."
        else:
            status, reason = "candidate", channel["reason"] + f" ({coverage_text})."
        option = {"label": channel["label"], "status": status, "reason": reason,
                  "required_external_checks": list(channel["checks"])}
        for row in rows:
            estimate = _matching_price(row, channel["id"], variety_text, grade_key, as_of, policy["required_price_adjustments"])
            if estimate is not None and status != "unavailable":
                option["price_estimate"] = estimate
                break
        options.append(option)

    if safety:
        options.append({"label": policy["reject"]["hold_label"], "status": "requires_check",
                        "reason": "Hold the lot pending qualified inspection/testing; image evidence cannot establish safety.",
                        "required_external_checks": list(policy["reject"]["hold_checks"])})
    elif grade_key == "Reject":
        if not safety:
            options.append({"label": policy["reject"]["feed_label"], "status": "requires_check",
                            "reason": "Reject grade may have a feed alternative, but image evidence cannot validate feed safety.",
                            "required_external_checks": list(policy["reject"]["feed_checks"])})
    return {"grade": grade_key, "variety": variety_text, "options": options,
            "safety_exclusions": safety}
