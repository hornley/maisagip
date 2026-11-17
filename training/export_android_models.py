"""Export the production corn models into ONNX assets for the offline Android app."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from backend.app import config


def _class_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in range(len(names))]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    raise ValueError("Checkpoint class names must be a list or contiguous integer-keyed mapping.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _onnx_metadata(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)

    def shape(value_info):
        dims = value_info.type.tensor_type.shape.dim
        return [dim.dim_value if dim.dim_value else -1 for dim in dims]

    inputs = list(model.graph.input)
    if len(inputs) != 1:
        raise ValueError(f"{path.name} must have exactly one model input; found {len(inputs)}.")
    return {
        "input_name": inputs[0].name,
        "input_shape": shape(inputs[0]),
        "output_names": [output.name for output in model.graph.output],
        "output_shapes": [shape(output) for output in model.graph.output],
    }


def _export_yolo(checkpoint: Path, output: Path, *, imgsz: int, opset: int) -> dict[str, Any]:
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="maisagip-onnx-") as temporary:
        temporary_root = Path(temporary)
        temporary_checkpoint = temporary_root / checkpoint.name
        shutil.copy2(checkpoint, temporary_checkpoint)
        model = YOLO(str(temporary_checkpoint))
        class_names = _class_names(model.names)
        exported = Path(model.export(
            format="onnx",
            imgsz=imgsz,
            batch=1,
            dynamic=False,
            simplify=False,
            nms=False,
            half=False,
            device="cpu",
            opset=opset,
        ))
        if not exported.is_file():
            raise FileNotFoundError(f"Ultralytics did not create an ONNX file for {checkpoint}.")
        shutil.copy2(exported, output)
    metadata = _onnx_metadata(output)
    return {**metadata, "class_names": class_names}


def _export_classifier(checkpoint: Path, output: Path, *, opset: int) -> dict[str, Any]:
    model = models.efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, len(config.VARIETY_CLASSES)),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 448, 448),
        str(output),
        input_names=["images"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    metadata = _onnx_metadata(output)
    return {
        **metadata,
        "class_names": list(config.VARIETY_CLASSES),
        "input_size": 448,
        "resize_short_edge_scale": 1.1,
        "center_crop": 448,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def export_models(weights_dir: Path, output_dir: Path, opset: int = 17) -> Path:
    checkpoints = {
        "segmenter": (weights_dir / "best.pt", "corn_ear_segmenter.onnx"),
        "classifier": (weights_dir / "efficientnetv2_s_corn.pt", "corn_variety_classifier.onnx"),
        "detector": (weights_dir / "yolo11s_colab_best_2026-10-08.pt", "corn_defect_detector.onnx"),
    }
    for path, _ in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained model checkpoint: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter_path = output_dir / checkpoints["segmenter"][1]
    classifier_path = output_dir / checkpoints["classifier"][1]
    detector_path = output_dir / checkpoints["detector"][1]
    segmenter = _export_yolo(checkpoints["segmenter"][0], segmenter_path, imgsz=640, opset=opset)
    classifier = _export_classifier(checkpoints["classifier"][0], classifier_path, opset=opset)
    detector = _export_yolo(checkpoints["detector"][0], detector_path, imgsz=640, opset=opset)

    for model, model_path in (
        (segmenter, segmenter_path),
        (classifier, classifier_path),
        (detector, detector_path),
    ):
        model["file"] = model_path.name
        model["sha256"] = _sha256(model_path)

    segmenter.update({"input_size": 640, "confidence_threshold": 0.25, "nms_iou": 0.7})
    classifier["input_size"] = 448
    detector.update({
        "input_size": 640,
        "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
        "nms_iou": config.DETECTOR_NMS_IOU,
        "canonical_class_names": [config.canonical_defect_class(name) for name in detector["class_names"]],
    })
    manifest = {
        "schema_version": 1,
        "models": {"segmenter": segmenter, "classifier": classifier, "detector": detector},
    }
    manifest_path = output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the three trained models for Android ONNX Runtime.")
    parser.add_argument("--weights-dir", type=Path, default=config.WEIGHTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("android/app/src/main/assets/models"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    print(export_models(args.weights_dir, args.output_dir, args.opset))


if __name__ == "__main__":
    main()
