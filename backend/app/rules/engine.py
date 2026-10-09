import json
from functools import lru_cache
from pathlib import Path

from .. import config

DEFAULT_RULES_PATH = Path(__file__).parent / "config_rules.json"
LOW_DEFECT_CONFIDENCE = 0.35


@lru_cache(maxsize=1)
def load_rules(rules_path=None):
    path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class GradeEngine:
    def __init__(self, rules=None):
        self.rules = rules if rules is not None else load_rules()

    def _assign_grade(self, coverage):
        grade_order = self.rules["grade_order"]
        grades = self.rules["grades"]
        reasons = []
        for key in grade_order:
            spec = grades[key]
            if "coverage_at_most" in spec:
                passes = coverage <= spec["coverage_at_most"]
                threshold = spec["coverage_at_most"]
                criterion = f"at most {threshold * 100:g}%"
            else:
                passes = coverage < spec["coverage_less_than"]
                threshold = spec["coverage_less_than"]
                criterion = f"below {threshold * 100:g}%"
            coverage_text = f"{coverage * 100:.2f}".rstrip("0").rstrip(".") + "%"
            if passes:
                reasons.append(f"Meets {spec['label']} criteria: {coverage_text} coverage is {criterion}.")
                return key, spec["label"], reasons
            reasons.append(f"Fails {spec['label']}: {coverage_text} coverage is not {criterion}.")
        reject_threshold = grades[grade_order[-1]]["coverage_less_than"] * 100
        reasons.append(f"Assigned Reject: coverage is {coverage_text}, at or above {reject_threshold:g}%.")
        return "Reject", self.rules["below_grade_label"], reasons

    def _choose_utilization(self, coverage, severe_present):
        for rule in self.rules["utilization"]:
            condition = rule["condition"]
            matched = False
            if condition == "coverage_ge":
                matched = coverage >= rule["value"]
            elif condition == "coverage_lt":
                matched = coverage < rule["value"]
            elif condition == "any_severe":
                matched = severe_present
            elif condition == "always":
                matched = True
            if matched:
                return rule["recommend"], rule["reason"]
        return "reject", "No applicable utilization rule."

    def evaluate(
        self,
        variety_conf,
        coverage,
        severe_present,
        completeness,
        defect_confs=(),
        confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
        variety_disagreement=False,
    ):
        grade, grade_label, grade_reasons = self._assign_grade(coverage)
        utilization, utilization_reason = self._choose_utilization(coverage, severe_present)

        if defect_confs:
            overall = min(variety_conf, sum(defect_confs) / len(defect_confs))
        else:
            overall = variety_conf
        overall = round(min(float(overall), 0.99), 4)

        low_conf = variety_conf < confidence_threshold
        low_defect_conf = any(c < LOW_DEFECT_CONFIDENCE for c in defect_confs)
        needs_reinspection = low_conf or low_defect_conf or variety_disagreement

        return {
            "grade": grade,
            "grade_label": grade_label,
            "grade_reasons": grade_reasons,
            "defect_coverage": round(float(coverage), 4),
            "severe_present": bool(severe_present),
            "utilization": {"recommendation": utilization, "reason": utilization_reason},
            "confidence": overall,
            "needs_reinspection": bool(needs_reinspection),
        }
