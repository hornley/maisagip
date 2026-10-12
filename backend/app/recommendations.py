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
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypedDict


CONFIG_PATH = Path(__file__).parent / "rules" / "config_recommendations.json"
GRADE_CHANNELS = MappingProxyType({
    "ExtraClass": frozenset({"local_market", "export", "premium_sale", "food_processing_canned"}),
    "ClassI": frozenset({"local_market", "premium_sale", "food_processing_canned"}),
    "ClassII": frozenset({"local_market"}),
    "Reject": frozenset({"poultry_feed"}),
})
SAFETY_EXCLUSIONS = frozenset({"mold", "ear_decay"})
CHANNEL_LABELS = MappingProxyType({
    "local_market": "Local Market",
    "export": "Export",
    "premium_sale": "Premium Sale",
    "food_processing_canned": "Processed/Canned",
    "poultry_feed": "Poultry Feed",
})
CHANNEL_ORDER = tuple(CHANNEL_LABELS)
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
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid recommendation config: invalid JSON") from exc
    if not isinstance(config, Mapping):
        raise ValueError("Invalid recommendation config: root must be an object")
    adjustments = config.get("required_price_adjustments")
    if (not isinstance(adjustments, list) or not adjustments or
            any(not isinstance(value, str) or not value.strip() for value in adjustments)):
        raise ValueError(
            "Invalid recommendation config: required_price_adjustments must be a non-empty array "
            "of non-empty strings"
        )
    if not isinstance(config.get("rates"), list):
        raise ValueError("Invalid recommendation config: rates must be an array")
    return config


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _defect_view(defects: Any) -> tuple[set[str], float | None]:
    """Read only explicit visual defect names and explicit coverage."""
    if isinstance(defects, Mapping):
        if isinstance(defects.get("class"), str) and not ({"classes", "defects"} & defects.keys()):
            raw_classes = [defects]
        else:
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
    low_number, high_number = _finite_number(low), _finite_number(high)
    if low_number is None or high_number is None or low_number < 0 or high_number < low_number:
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
        factor_number = _finite_number(factor)
        if factor_number is None or factor_number <= 0:
            return None
        multiplier *= factor_number
        if not math.isfinite(multiplier):
            return None
    adjusted_low, adjusted_high = low_number * multiplier, high_number * multiplier
    if not math.isfinite(adjusted_low) or not math.isfinite(adjusted_high):
        return None
    source = record.get("source")
    if not isinstance(source, str) or not source.strip():
        return None
    return {
        "market": channel,
        "currency": "PHP",
        "unit": "kg",
        "low": round(adjusted_low, 2),
        "high": round(adjusted_high, 2),
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
    operational_config = _config()
    grade_key = _grade_key(grade)
    variety_text = variety if isinstance(variety, str) else ""
    defect_names, _ = _defect_view(defects)
    safety = sorted(defect_names.intersection(SAFETY_EXCLUSIONS))
    eligible_ids = GRADE_CHANNELS.get(grade_key, frozenset())
    if safety and grade_key != "Reject":
        eligible_ids = eligible_ids.intersection({"local_market"})
    as_of = _as_date(inspection_date) or date.today()
    rows = operational_config["rates"] if price_data is None else (
        price_data.get("rates", []) if isinstance(price_data, Mapping) else price_data
    )
    rows = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []

    options = []
    for channel_id in CHANNEL_ORDER:
        if channel_id not in eligible_ids:
            continue
        option = {"label": CHANNEL_LABELS[channel_id]}
        for row in rows:
            estimate = _matching_price(
                row, channel_id, variety_text, grade_key, as_of,
                operational_config["required_price_adjustments"],
            )
            if estimate is not None:
                option["price_estimate"] = estimate
                break
        options.append(option)

    return {"grade": grade_key, "variety": variety_text, "options": options,
            "safety_exclusions": safety}
