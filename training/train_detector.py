import argparse

from backend.app import config

from make_dataset_split import RATIOS


def build_data_yaml(out_path, root, nc, names):
    segments = [f"path: {root.as_posix()}"]
    for split in RATIOS:
        segments.append(f"{split}: images/{split}")
    segments.append(f"nc: {nc}")
    segments.append(f"names: {names}")
    out_path.write_text("\n".join(segments) + "\n", encoding="utf-8")


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
    model.train(data=str(data_yaml), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, name="corn_detector")

    best_pt = next(config.PROJECT_ROOT.rglob("corn_detector/weights/best.pt"), None)
    if best_pt is not None:
        config.DETECTOR_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        config.DETECTOR_WEIGHTS.write_bytes(best_pt.read_bytes())
        print(f"Saved detector weights to {config.DETECTOR_WEIGHTS}")


if __name__ == "__main__":
    main()