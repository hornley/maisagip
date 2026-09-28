import argparse
from pathlib import Path
from typing import Any, Iterable

from backend.app import config
from training.yolo_comparison import build_data_yaml


def _path_attribute(objects: Iterable[Any], names: Iterable[str]) -> Path | None:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                try:
                    return Path(value)
                except TypeError:
                    continue
    return None


def _best_weights_from_training(
    train_result: Any, model: Any, project: Path, name: str
) -> Path | None:
    """Find the newest best checkpoint produced by this training invocation."""
    trainer = getattr(model, "trainer", None)
    sources = (train_result, trainer, model)
    save_dirs = []
    for source in sources:
        save_dir = _path_attribute((source,), ("save_dir",))
        if save_dir is not None and save_dir not in save_dirs:
            save_dirs.append(save_dir)

    candidates = []
    for source in sources:
        best = _path_attribute((source,), ("best", "best_weights", "best_weights_path"))
        if best is not None:
            candidates.append(best)
    candidates.extend(save_dir / "weights" / "best.pt" for save_dir in save_dirs)
    candidates.append(project / name / "weights" / "best.pt")
    if project.exists():
        candidates.extend(project.glob(f"{name}*/weights/best.pt"))

    existing = []
    for candidate in candidates:
        if candidate.is_file() and candidate not in existing:
            existing.append(candidate)
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime_ns)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11n for corn defect detection.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default="yolo11n.pt")
    args = parser.parse_args()

    data_root = config.DATA_DIR / "detector"
    data_yaml = data_root / "data.yaml"
    build_data_yaml(data_yaml, data_root, len(config.DEFECT_CLASSES), config.DEFECT_CLASSES)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    project = config.PROJECT_ROOT / "runs" / "detect"
    train_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(project),
        name="corn_detector",
        exist_ok=False,
    )

    best_pt = _best_weights_from_training(train_result, model, project, "corn_detector")
    if best_pt is not None and best_pt.is_file():
        config.DETECTOR_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        config.DETECTOR_WEIGHTS.write_bytes(best_pt.read_bytes())
        print(f"Saved detector weights to {config.DETECTOR_WEIGHTS}")


if __name__ == "__main__":
    main()
