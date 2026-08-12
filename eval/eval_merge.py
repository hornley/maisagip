import argparse
import csv
from pathlib import Path

from backend.app import config
from backend.app.pipeline import inspect_ear

GRADE_MAP = {
    "belowclassii": "Below Class II",
    "below": "Below Class II",
    "reject": "Below Class II",
    "classi": "Class I",
    "class1": "Class I",
    "classii": "Class II",
    "class2": "Class II",
    "extra": "Extra Class",
}


def canonical_grade(value):
    if value is None:
        return "Below Class II"
    return GRADE_MAP.get("".join(str(value).lower().split()), str(value).strip())


def _ear_image_paths(ear_id, search_root):
    root = Path(search_root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir() if p.stem.startswith(ear_id) and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate merge correctness vs a ground-truth ear CSV.")
    parser.add_argument("--csv", default="data/raw/ground_truth.csv")
    parser.add_argument("--images", default="data/raw/detector/images")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"ground truth CSV not found: {args.csv}")
        return

    cov_errors = []
    grade_agreement = 0
    total = 0

    with Path(args.csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ear_id = row["ear_id"]
            files = _ear_image_paths(ear_id, args.images)
            if not files:
                print(f"  skip {ear_id}: no images found")
                continue
            report = inspect_ear([p.read_bytes() for p in files])
            model_coverage = report["defect_coverage"]
            true_coverage = float(row.get("coverage", 0.0) or 0.0)
            cov_errors.append(abs(model_coverage - true_coverage))
            expected = canonical_grade(row.get("grade", ""))
            actual = canonical_grade(report["grade"]["grade_label"])
            if expected == actual:
                grade_agreement += 1
            else:
                print(f"  grade mismatch {ear_id}: human={expected} model={actual}")
            total += 1

    if total:
        mae = sum(cov_errors) / len(cov_errors)
        print(f"ears evaluated: {total}")
        print(f"coverage mean absolute error: {mae:.4f}")
        print(f"grade agreement: {grade_agreement}/{total} ({grade_agreement / total:.2%})")
    else:
        print("no ears evaluated")


if __name__ == "__main__":
    main()