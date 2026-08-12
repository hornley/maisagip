import argparse
import time
from pathlib import Path

from backend.app import config


def main():
    parser = argparse.ArgumentParser(description="Evaluate the corn defect detector (mAP, IoU, latency).")
    parser.add_argument("--weights", default=str(config.DETECTOR_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"weights not found: {args.weights}")
        return

    from ultralytics import YOLO

    model = YOLO(args.weights)
    data_yaml = config.DATA_DIR / "detector" / "data.yaml"
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz)

    print(f"mAP@0.5        = {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95   = {metrics.box.map:.4f}")
    print(f"precision      = {metrics.box.mp:.4f}")
    print(f"recall         = {metrics.box.mr:.4f}")

    test_dir = config.DATA_DIR / "detector" / "images" / "test"
    images = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        print("no test images for latency measurement")
        return

    times = []
    for img in images:
        start = time.perf_counter()
        model.predict(str(img), imgsz=args.imgsz, verbose=False)
        times.append(time.perf_counter() - start)
    avg_ms = sum(times) / len(times) * 1000
    print(f"inference time (mean over {len(images)} images) = {avg_ms:.1f} ms")


if __name__ == "__main__":
    main()