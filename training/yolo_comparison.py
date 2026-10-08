import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from backend.app import config
from training.make_dataset_split import RATIOS


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    weights: str


YOLO_MODELS = (
    ModelSpec("yolov7-tiny", "yolov7", "yolov7-tiny.pt"),
    ModelSpec("yolov7", "yolov7", "yolov7.pt"),
    ModelSpec("yolov7x", "yolov7", "yolov7x.pt"),
    ModelSpec("yolov8n", "ultralytics", "yolov8n.pt"),
    ModelSpec("yolov8s", "ultralytics", "yolov8s.pt"),
    ModelSpec("yolov8m", "ultralytics", "yolov8m.pt"),
    ModelSpec("yolo11s", "ultralytics", "yolo11s.pt"),
)

YOLOV7_CONFIGS = {
    "yolov7-tiny": "cfg/training/yolov7-tiny.yaml",
    "yolov7": "cfg/training/yolov7.yaml",
    "yolov7x": "cfg/training/yolov7x.yaml",
}

YOLOV7_TORCH_WEIGHTS_ONLY_ENV = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
YOLOV7_NUMPY_COMPATIBILITY = "numpy.trapz -> numpy.trapezoid when needed"


def model_names():
    return [model.name for model in YOLO_MODELS]


def build_data_yaml(out_path, root, nc=None, names=None):
    """Write the detector dataset YAML shared by all YOLO training paths."""
    root = Path(root).resolve()
    out_path = Path(out_path)
    nc = len(config.DEFECT_CLASSES) if nc is None else nc
    names = config.DEFECT_CLASSES if names is None else names

    segments = [f"path: {root.as_posix()}"]
    for split in RATIOS:
        segments.append(f"{split}: images/{split}")
    segments.append(f"nc: {nc}")
    segments.append(f"names: {names}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(segments) + "\n", encoding="utf-8")


def build_yolov7_data_yaml(out_path, root, nc=None, names=None):
    """Write the absolute-path dataset YAML required by native YOLOv7."""
    root = Path(root).expanduser().resolve()
    out_path = Path(out_path)
    nc = len(config.DEFECT_CLASSES) if nc is None else nc
    names = config.DEFECT_CLASSES if names is None else names

    segments = []
    for split in RATIOS:
        segments.append(f"{split}: {(root / 'images' / split).as_posix()}")
    segments.extend((f"nc: {nc}", f"names: {names}"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(segments) + "\n", encoding="utf-8")


def _yolov7_data_yaml_path(output_root: Path) -> Path:
    """Return the isolated generated YAML path without reserving a model run."""
    return output_root / "generated" / "yolov7-data.yaml"


def _yolov7_numpy_shim_path(output_root: Path) -> Path:
    """Return the isolated sitecustomize path used by native YOLOv7."""
    return output_root / "generated" / "sitecustomize.py"


def _write_yolov7_numpy_shim(output_root: Path) -> Path:
    """Generate a conditional NumPy compatibility shim for native children."""
    shim_path = _yolov7_numpy_shim_path(output_root)
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    shim_path.write_text(
        """\
\"\"\"Compatibility aliases for trusted native YOLOv7 subprocesses.\"\"\"

import numpy as np

if not hasattr(np, \"trapz\") and hasattr(np, \"trapezoid\"):
    np.trapz = np.trapezoid
""",
        encoding="utf-8",
    )
    return shim_path


def _yolov7_numpy_compatibility(output_root: Path) -> dict[str, Any]:
    shim_path = _yolov7_numpy_shim_path(output_root)
    return {
        "name": YOLOV7_NUMPY_COMPATIBILITY,
        "sitecustomize": str(shim_path),
        "behavior": "aliases numpy.trapz to numpy.trapezoid only when trapz is absent",
    }


def _yolov7_native_environment(output_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Build the shared environment and compatibility record for native children."""
    shim_path = _write_yolov7_numpy_shim(output_root)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(shim_path.parent), existing_pythonpath) if part
    )
    env[YOLOV7_TORCH_WEIGHTS_ONLY_ENV] = "1"
    return env, _yolov7_numpy_compatibility(output_root)


RESULT_FIELDS = (
    "model",
    "run_dir",
    "best_weights",
    "map50",
    "map",
    "precision",
    "recall",
)


def model_spec(name: str) -> ModelSpec:
    """Return a catalog entry, with an actionable error for unknown names."""
    for spec in YOLO_MODELS:
        if spec.name == name:
            return spec
    available = ", ".join(model_names())
    raise ValueError(f"Unknown model '{name}'. Choose one of: {available}")


def selected_models(value: str | None) -> list[ModelSpec]:
    if value is None:
        return list(YOLO_MODELS)
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--models must contain at least one model name")
    return [model_spec(name) for name in names]


def yolov7_config(model_name: str) -> str:
    """Return the native YOLOv7 config path for a catalog model."""
    try:
        return YOLOV7_CONFIGS[model_name]
    except KeyError as exc:
        available = ", ".join(YOLOV7_CONFIGS)
        raise ValueError(
            f"No native YOLOv7 config is defined for '{model_name}'. "
            f"Choose one of: {available}"
        ) from exc


def _metric(metrics: Any, name: str) -> float | None:
    """Read a metric without turning unavailable values into made-up scores."""
    try:
        value = getattr(metrics, name)
    except (AttributeError, TypeError):
        return None
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        return None
    return numeric_value


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer from 0 through 4294967295")
    if not 0 <= seed <= 4_294_967_295:
        raise ValueError(
            f"seed must be from 0 through 4294967295; received {seed}"
        )
    return seed


def _path_attribute(objects: Iterable[Any], names: Iterable[str]) -> Path | None:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return Path(value)
    return None


def _training_paths(
    train_result: Any, model: Any, output_root: Path, spec: ModelSpec
) -> tuple[Path, Path]:
    """Resolve paths from Ultralytics so suffixed reruns are reported correctly."""
    trainer = getattr(model, "trainer", None)
    sources = (train_result, trainer, model)
    run_dir = _path_attribute(sources, ("save_dir",))
    if run_dir is None:
        run_dir = output_root / spec.name

    best_weights = _path_attribute(
        sources, ("best", "best_weights", "best_weights_path")
    )
    if best_weights is None:
        best_weights = run_dir / "weights" / "best.pt"
    return run_dir, best_weights


def run_ultralytics_model(
    spec: ModelSpec,
    data_yaml: str | Path,
    output_root: str | Path,
    *,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    device: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate one Ultralytics model in isolated output directories."""
    seed = _validate_seed(seed)
    if spec.family != "ultralytics":
        raise ValueError(f"Model '{spec.name}' is not an Ultralytics model")

    # Keep this import lazy: planning and --dry-run must work without ML packages.
    from ultralytics import YOLO

    data_yaml = Path(data_yaml)
    output_root = Path(output_root)
    model = YOLO(spec.weights)

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "seed": seed,
        "project": str(output_root),
        "name": spec.name,
        "exist_ok": False,
    }
    if device is not None:
        train_kwargs["device"] = device
    train_result = model.train(**train_kwargs)

    val_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": "test",
        "imgsz": imgsz,
        "project": str(output_root),
        "name": f"{spec.name}-test",
        "exist_ok": False,
    }
    if device is not None:
        val_kwargs["device"] = device
    val_result = model.val(**val_kwargs)

    run_dir, best_weights = _training_paths(train_result, model, output_root, spec)
    metrics = getattr(val_result, "box", None)
    metric_values = {}
    for result_name, metric_name in (
        ("map50", "map50"),
        ("map", "map"),
        ("precision", "mp"),
        ("recall", "mr"),
    ):
        value = _metric(metrics, metric_name)
        if value is None:
            warnings.warn(
                f"Ultralytics metric '{metric_name}' was unavailable or invalid "
                f"for {spec.name}; recording None.",
                RuntimeWarning,
            )
        metric_values[result_name] = value
    return {
        "model": spec.name,
        "run_dir": str(run_dir),
        "best_weights": str(best_weights),
        **metric_values,
    }


def _validate_yolov7_checkout(
    yolov7_root: str | Path | None,
    specs: Iterable[ModelSpec],
    yolov7_weights_dir: str | Path | None = None,
) -> Path:
    specs = list(specs)
    if yolov7_root is None:
        names = ", ".join(spec.name for spec in specs)
        raise RuntimeError(
            "YOLOv7 models require the official YOLOv7 checkout. "
            f"Pass --yolov7-root /path/to/yolov7 to run: {names}. "
            "No training was started."
        )

    root = Path(yolov7_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(
            f"YOLOv7 checkout not found at '{root}'. The official YOLOv7 "
            "checkout is required via --yolov7-root. No training was started."
        )

    required = [root / "train.py", root / "test.py"]
    required.extend(root / yolov7_config(spec.name) for spec in specs)
    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            "YOLOv7 checkout is missing required files: "
            f"{missing_text}. Check --yolov7-root and use an official checkout. "
            "No training was started."
        )

    if yolov7_weights_dir is not None:
        weights_dir = Path(yolov7_weights_dir).expanduser().resolve()
        missing_weights = [
            weights_dir / spec.weights
            for spec in specs
            if not (weights_dir / spec.weights).is_file()
        ]
        if missing_weights:
            missing_text = ", ".join(str(path) for path in missing_weights)
            raise RuntimeError(
                "YOLOv7 local weights were requested with "
                f"--yolov7-weights-dir, but these files are unavailable: "
                f"{missing_text}. Download the matching weights or omit the "
                "weights-dir option to use the native standard filename. "
                "No training was started."
            )
    return root


def _yolov7_weight_argument(
    spec: ModelSpec, yolov7_weights_dir: str | Path | None
) -> str:
    if yolov7_weights_dir is None:
        return spec.weights
    return str((Path(yolov7_weights_dir).expanduser() / spec.weights).resolve())


def _python_executable(yolov7_python: str | Path | None) -> str:
    return str(yolov7_python) if yolov7_python is not None else sys.executable


def _effective_yolov7_device(device: str | None) -> str | None:
    """Map devices unsupported by official YOLOv7 to a usable native device."""
    return "cpu" if device == "mps" else device


def build_yolov7_train_command(
    spec: ModelSpec,
    yolov7_root: str | Path,
    data_yaml: str | Path,
    output_root: str | Path,
    *,
    run_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    weights: str,
    yolov7_python: str | Path | None = None,
    device: str | None = None,
) -> list[str]:
    """Build an inspectable command for the official YOLOv7 train.py."""
    if spec.family != "yolov7":
        raise ValueError(f"Model '{spec.name}' is not a YOLOv7 model")
    root = Path(yolov7_root)
    command = [
        _python_executable(yolov7_python),
        str(root / "train.py"),
        "--data",
        str(Path(data_yaml).resolve()),
        "--cfg",
        str(root / yolov7_config(spec.name)),
        "--weights",
        weights,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch),
        "--img-size",
        str(imgsz),
        "--project",
        str(Path(output_root).resolve()),
        "--name",
        run_name,
    ]
    if device is not None:
        command.extend(("--device", device))
    return command


def build_yolov7_test_command(
    spec: ModelSpec,
    yolov7_root: str | Path,
    data_yaml: str | Path,
    output_root: str | Path,
    *,
    run_name: str,
    weights: str,
    batch: int,
    imgsz: int,
    yolov7_python: str | Path | None = None,
    device: str | None = None,
) -> list[str]:
    """Build an inspectable command for YOLOv7's held-out test split."""
    if spec.family != "yolov7":
        raise ValueError(f"Model '{spec.name}' is not a YOLOv7 model")
    root = Path(yolov7_root)
    command = [
        _python_executable(yolov7_python),
        str(root / "test.py"),
        "--data",
        str(Path(data_yaml).resolve()),
        "--weights",
        weights,
        "--batch-size",
        str(batch),
        "--img-size",
        str(imgsz),
        "--task",
        "test",
        "--project",
        str(Path(output_root).resolve()),
        "--name",
        f"{run_name}-test",
    ]
    if device is not None:
        command.extend(("--device", device))
    return command


def _unique_run_dir(output_root: Path, spec: ModelSpec) -> Path:
    while True:
        candidate = output_root / f"{spec.name}-{uuid.uuid4().hex[:12]}"
        if not candidate.exists():
            return candidate


def _native_run_dirs(output_root: Path, run_name: str) -> set[Path]:
    pattern = re.compile(rf"^{re.escape(run_name)}(?P<suffix>\d*)$")
    candidates = set()
    if not output_root.is_dir():
        return candidates
    for candidate in output_root.iterdir():
        if not candidate.is_dir():
            continue
        match = pattern.match(candidate.name)
        if match is None:
            continue
        candidates.add(candidate)
    return candidates


def _resolve_native_run_dir(
    output_root: Path,
    run_name: str,
    intended_run_dir: Path,
    preexisting_run_dirs: set[Path],
) -> Path:
    """Resolve the run created by this train invocation, not stale output."""
    run_dirs = _native_run_dirs(output_root, run_name)
    if intended_run_dir not in preexisting_run_dirs and intended_run_dir in run_dirs:
        return intended_run_dir

    new_run_dirs = run_dirs - preexisting_run_dirs
    if len(new_run_dirs) == 1:
        return next(iter(new_run_dirs))
    if not new_run_dirs:
        raise RuntimeError(
            "Could not identify the YOLOv7 run directory created by the current "
            f"train command for '{run_name}': no new matching run directory was "
            "created. Refusing to use an existing output directory."
        )
    raise RuntimeError(
        "Could not identify the YOLOv7 run directory created by the current "
        f"train command for '{run_name}'; multiple new matching directories "
        f"were found: {', '.join(sorted(str(path) for path in new_run_dirs))}."
    )


def _write_native_output(
    run_dir: Path,
    phase: str,
    command: list[str],
    stdout: str,
    stderr: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{phase}-command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    (run_dir / f"{phase}-stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / f"{phase}-stderr.txt").write_text(stderr, encoding="utf-8")


def _write_run_metadata(
    run_dir: Path,
    *,
    spec: ModelSpec,
    seed: int,
    data_yaml: Path,
    output_root: Path,
    train_command: list[str],
    test_command: list[str],
    requested_device: str | None,
    effective_yolov7_device: str | None,
    compatibility: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": spec.name,
        "seed": seed,
        "environment": {
            "PYTHONHASHSEED": str(seed),
            YOLOV7_TORCH_WEIGHTS_ONLY_ENV: "1",
        },
        "data_yaml": str(data_yaml.resolve()),
        "output_root": str(output_root),
        "requested_device": requested_device,
        "effective_yolov7_device": effective_yolov7_device,
        "compatibility": compatibility,
        "train_command": train_command,
        "test_command": test_command,
    }
    (run_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _run_native_command(
    command: list[str],
    root: Path,
    run_dir: Path,
    phase: str,
    env: dict[str, str],
    *,
    defer_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except OSError as exc:
        _write_native_output(run_dir, phase, command, "", str(exc))
        raise RuntimeError(
            f"Could not start YOLOv7 {phase} command: {exc}. "
            f"Inspect {run_dir / f'{phase}-command.txt'} and verify "
            "--yolov7-python and the checkout."
        ) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if not defer_output:
        _write_native_output(run_dir, phase, command, stdout, stderr)
    if completed.returncode != 0 and not defer_output:
        _raise_native_command_failure(completed, run_dir, phase)
    return completed


def _raise_native_command_failure(
    completed: subprocess.CompletedProcess[str], run_dir: Path, phase: str
) -> None:
    seed_hint = (
        " The requested seed was exported as PYTHONHASHSEED, but official "
        "YOLOv7 keeps its internal seed hardcoded."
        if phase == "train"
        else ""
    )
    weights_hint = (
        " If the standard initial weight filename is unavailable, provide "
        "--yolov7-weights-dir with a local matching file."
        if phase == "train"
        else ""
    )
    weights_only_hint = (
        " The compatibility environment sets "
        f"{YOLOV7_TORCH_WEIGHTS_ONLY_ENV}=1; use it only with trusted official "
        "YOLOv7 checkpoints."
    )
    raise RuntimeError(
        f"YOLOv7 {phase} command failed with exit code "
        f"{completed.returncode}.{seed_hint}{weights_hint}{weights_only_hint} "
        "Raw output was preserved in "
        f"{run_dir / f'{phase}-stdout.txt'} and "
        f"{run_dir / f'{phase}-stderr.txt'}."
    )


def _parse_yolov7_metrics(output: str) -> dict[str, float] | None:
    """Parse the standard YOLOv7 aggregate row: P, R, mAP@.5, mAP@.5:.95."""
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) != 7 or fields[0].lower() != "all":
            continue
        try:
            image_count = int(fields[1])
            label_count = int(fields[2])
            precision, recall, map50, map_value = (
                float(value) for value in fields[3:]
            )
        except ValueError:
            continue
        if image_count < 0 or label_count < 0:
            continue
        values = (precision, recall, map50, map_value)
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
        ):
            continue
        return {
            "map50": map50,
            "map": map_value,
            "precision": precision,
            "recall": recall,
        }
    return None


def run_yolov7_model(
    spec: ModelSpec,
    data_yaml: str | Path,
    output_root: str | Path,
    *,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    yolov7_root: str | Path,
    yolov7_python: str | Path | None = None,
    yolov7_weights_dir: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train and test one model through an official YOLOv7 checkout."""
    seed = _validate_seed(seed)
    output_root = Path(output_root).expanduser().resolve()
    if spec.family != "yolov7":
        raise ValueError(f"Model '{spec.name}' is not a YOLOv7 model")
    root = _validate_yolov7_checkout(
        yolov7_root, [spec], yolov7_weights_dir=yolov7_weights_dir
    )
    effective_device = _effective_yolov7_device(device)
    if device == "mps":
        warnings.warn(
            "Official YOLOv7 cannot use MPS on this system; mapping requested "
            "device mps to CPU for native YOLOv7 train and test commands.",
            RuntimeWarning,
        )
    run_dir = _unique_run_dir(output_root, spec)
    run_name = run_dir.name
    preexisting_run_dirs = _native_run_dirs(output_root, run_name)
    weights = _yolov7_weight_argument(spec, yolov7_weights_dir)
    train_command = build_yolov7_train_command(
        spec,
        root,
        data_yaml,
        output_root,
        run_name=run_name,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        seed=seed,
        weights=weights,
        yolov7_python=yolov7_python,
        device=effective_device,
    )
    native_env, compatibility = _yolov7_native_environment(output_root)
    native_env["PYTHONHASHSEED"] = str(seed)
    warnings.warn(
        "Official YOLOv7 train.py does not accept --seed; requested seed "
        f"{seed} is recorded in run-metadata.json and exported as "
        "PYTHONHASHSEED, but YOLOv7's internal seed remains hardcoded. "
        f"{YOLOV7_TORCH_WEIGHTS_ONLY_ENV}=1 is also exported for compatibility "
        "with trusted official YOLOv7 checkpoints only.",
        RuntimeWarning,
    )
    train_result = _run_native_command(
        train_command,
        root,
        run_dir,
        "train",
        native_env,
        defer_output=True,
    )
    run_dir = _resolve_native_run_dir(
        output_root, run_name, run_dir, preexisting_run_dirs
    )
    best_weights = run_dir / "weights" / "best.pt"
    test_command = build_yolov7_test_command(
        spec,
        root,
        data_yaml,
        output_root,
        run_name=run_dir.name,
        weights=str(best_weights),
        batch=batch,
        imgsz=imgsz,
        yolov7_python=yolov7_python,
        device=effective_device,
    )
    _write_run_metadata(
        run_dir,
        spec=spec,
        seed=seed,
        data_yaml=Path(data_yaml),
        output_root=output_root,
        train_command=train_command,
        test_command=test_command,
        requested_device=device,
        effective_yolov7_device=effective_device,
        compatibility=compatibility,
    )
    _write_native_output(
        run_dir,
        "train",
        train_command,
        train_result.stdout or "",
        train_result.stderr or "",
    )
    if train_result.returncode != 0:
        _raise_native_command_failure(train_result, run_dir, "train")
    test_result = _run_native_command(test_command, root, run_dir, "test", native_env)
    metrics = _parse_yolov7_metrics(
        (test_result.stdout or "") + "\n" + (test_result.stderr or "")
    )
    if metrics is None:
        warnings.warn(
            "Could not parse a standard aggregate metrics row from YOLOv7 "
            f"test output for {spec.name}; raw output is preserved in {run_dir}.",
            RuntimeWarning,
        )
        metrics = {key: None for key in ("map50", "map", "precision", "recall")}
    return {
        "model": spec.name,
        "run_dir": str(run_dir),
        "best_weights": str(best_weights),
        **metrics,
    }


def _run_model(
    spec: ModelSpec,
    data_yaml: Path,
    output_root: Path,
    *,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    device: str | None,
    yolov7_root: str | Path | None,
    yolov7_python: str | Path | None,
    yolov7_weights_dir: str | Path | None,
) -> dict[str, Any]:
    if spec.family == "ultralytics":
        return run_ultralytics_model(
            spec,
            data_yaml,
            output_root,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            seed=seed,
            device=device,
        )
    return run_yolov7_model(
        spec,
        data_yaml,
        output_root,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        seed=seed,
        yolov7_root=yolov7_root,
        yolov7_python=yolov7_python,
        yolov7_weights_dir=yolov7_weights_dir,
        device=device,
    )


def _write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _validate_comparison_output_root(output_root: str | Path) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    weights_root = Path(config.WEIGHTS_DIR).expanduser().resolve()
    if output_root == weights_root or weights_root in output_root.parents:
        raise ValueError(
            f"Comparison --output-root '{output_root}' must not equal or be "
            f"inside the production weights directory '{weights_root}'."
        )
    return output_root


def run_comparison(
    specs: Iterable[ModelSpec],
    data_yaml: str | Path,
    output_root: str | Path,
    *,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    device: str | None = None,
    yolov7_root: str | Path | None = None,
    yolov7_python: str | Path | None = None,
    yolov7_weights_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run selected models and publish summaries only after all runs finish."""
    output_root = _validate_comparison_output_root(output_root)
    seed = _validate_seed(seed)
    specs = list(specs)
    yolov7_specs = [spec for spec in specs if spec.family == "yolov7"]
    if yolov7_specs:
        # Preflight every v7 dependency before the first model starts. This is
        # important when a mixed run would otherwise train an Ultralytics model
        # before discovering a missing YOLOv7 checkout.
        _validate_yolov7_checkout(
            yolov7_root,
            yolov7_specs,
            yolov7_weights_dir=yolov7_weights_dir,
        )
    data_yaml = Path(data_yaml)
    model_data_yaml = data_yaml
    if yolov7_specs:
        model_data_yaml = _yolov7_data_yaml_path(output_root)
        build_yolov7_data_yaml(model_data_yaml, data_yaml.parent)
    rows = []
    for spec in specs:
        rows.append(
            _run_model(
                spec,
                model_data_yaml if spec.family == "yolov7" else data_yaml,
                output_root,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                seed=seed,
                device=device,
                yolov7_root=yolov7_root,
                yolov7_python=yolov7_python,
                yolov7_weights_dir=yolov7_weights_dir,
            )
        )
    _write_summary(rows, output_root)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare YOLO detector models.")
    parser.add_argument("--models", help="Comma-separated model names; defaults to all models")
    parser.add_argument("--data-root", default="data/detector", type=Path)
    parser.add_argument("--output-root", default="data/comparisons/yolo", type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device")
    parser.add_argument("--yolov7-root", type=Path)
    parser.add_argument("--yolov7-python", type=Path)
    parser.add_argument("--yolov7-weights-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _print_dry_run(
    specs: Iterable[ModelSpec],
    data_yaml: Path,
    output_root: Path,
    *,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    device: str | None,
    yolov7_root: Path | None,
    yolov7_python: Path | None,
    yolov7_weights_dir: Path | None,
) -> None:
    """Print complete planned invocations without importing ML packages."""
    output_root = Path(output_root).expanduser().resolve()
    for spec in specs:
        if spec.family == "yolov7":
            native_data_yaml = _yolov7_data_yaml_path(output_root)
            root = yolov7_root or Path("<yolov7-root>")
            effective_device = _effective_yolov7_device(device)
            weights = (
                _yolov7_weight_argument(spec, yolov7_weights_dir)
                if yolov7_weights_dir is not None
                else spec.weights
            )
            run_name = f"{spec.name}-DRY-RUN"
            train_command = build_yolov7_train_command(
                spec,
                root,
                native_data_yaml,
                output_root,
                run_name=run_name,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                seed=seed,
                weights=weights,
                yolov7_python=yolov7_python,
                device=effective_device,
            )
            test_command = build_yolov7_test_command(
                spec,
                root,
                native_data_yaml,
                output_root,
                run_name=run_name,
                weights=str(output_root / run_name / "weights" / "best.pt"),
                batch=batch,
                imgsz=imgsz,
                yolov7_python=yolov7_python,
                device=effective_device,
            )
            print(
                f"{spec.name}: requires official YOLOv7 checkout via "
                f"--yolov7-root; requested device={device!r}; "
                f"effective YOLOv7 device={effective_device!r}; "
                + (
                    "WARNING: official YOLOv7 cannot use MPS here; running on CPU; "
                    if device == "mps"
                    else ""
                )
                + f"seed metadata: PYTHONHASHSEED={seed}; "
                f"compatibility: {YOLOV7_TORCH_WEIGHTS_ONLY_ENV}=1 "
                "(trusted official YOLOv7 checkpoints only); "
                f"{YOLOV7_NUMPY_COMPATIBILITY}; "
                f"generated sitecustomize={_yolov7_numpy_shim_path(output_root)}; "
                f"native PYTHONPATH prepends={output_root / 'generated'}; "
                f"train: {shlex.join(train_command)}; "
                f"test: {shlex.join(test_command)}"
            )
            continue

        device_argument = f", device={device!r}" if device is not None else ""
        print(
            f"{spec.name}: ultralytics YOLO({spec.weights!r}).train("
            f"data={str(data_yaml)!r}, epochs={epochs}, batch={batch}, "
            f"imgsz={imgsz}, seed={seed}, project={str(output_root)!r}, "
            f"name={spec.name!r}, exist_ok=False{device_argument}); "
            f"val(split='test', "
            f"imgsz={imgsz}, project={str(output_root)!r}, "
            f"name={spec.name + '-test'!r}, exist_ok=False{device_argument})"
        )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        _validate_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        specs = selected_models(args.models)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        output_root = _validate_comparison_output_root(args.output_root)
    except ValueError as exc:
        parser.error(str(exc))

    data_yaml = args.data_root / "data.yaml"
    if args.dry_run:
        _print_dry_run(
            specs,
            data_yaml,
            output_root,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            seed=args.seed,
            device=args.device,
            yolov7_root=args.yolov7_root,
            yolov7_python=args.yolov7_python,
            yolov7_weights_dir=args.yolov7_weights_dir,
        )
        return

    # The YAML is shared with the production trainer, but comparison outputs stay
    # under output_root and never copy anything to config.DETECTOR_WEIGHTS.
    build_data_yaml(data_yaml, args.data_root)
    try:
        run_comparison(
            specs,
            data_yaml,
            output_root,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            seed=args.seed,
            device=args.device,
            yolov7_root=args.yolov7_root,
            yolov7_python=args.yolov7_python,
            yolov7_weights_dir=args.yolov7_weights_dir,
        )
    except (RuntimeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
