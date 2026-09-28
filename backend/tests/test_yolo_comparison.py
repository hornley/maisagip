import csv
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from backend.app import config
from training import yolo_comparison
from training.yolo_comparison import (
    ModelSpec,
    build_data_yaml,
    build_yolov7_data_yaml,
    build_yolov7_test_command,
    build_yolov7_train_command,
    model_names,
    run_comparison,
    run_ultralytics_model,
    run_yolov7_model,
    _parse_yolov7_metrics,
    yolov7_config,
)


def test_model_names_are_stable_and_complete():
    assert model_names() == [
        "yolov7-tiny",
        "yolov7",
        "yolov7x",
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolo11s",
    ]


def test_model_spec_is_immutable():
    spec = ModelSpec("example", "family", "weights.pt")

    try:
        spec.name = "changed"
    except AttributeError:
        pass
    else:
        raise AssertionError("ModelSpec must be immutable")


def test_build_data_yaml_writes_shared_absolute_dataset_definition(tmp_path):
    root = tmp_path / "detector"
    output = tmp_path / "comparison" / "data.yaml"

    build_data_yaml(output, root)

    content = output.read_text(encoding="utf-8")
    assert content == (
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(config.DEFECT_CLASSES)}\n"
        f"names: {config.DEFECT_CLASSES}\n"
    )
    assert content.endswith("\n")
    assert not (tmp_path / "comparison" / "weights").exists()


def test_build_yolov7_data_yaml_writes_absolute_split_paths(tmp_path):
    root = tmp_path / "detector"
    output = tmp_path / "comparison" / "generated" / "yolov7-data.yaml"

    build_yolov7_data_yaml(output, root)

    content = output.read_text(encoding="utf-8")
    assert content == (
        f"train: {(root / 'images/train').resolve().as_posix()}\n"
        f"val: {(root / 'images/val').resolve().as_posix()}\n"
        f"test: {(root / 'images/test').resolve().as_posix()}\n"
        f"nc: {len(config.DEFECT_CLASSES)}\n"
        f"names: {config.DEFECT_CLASSES}\n"
    )
    assert "path:" not in content
    assert output.parent.is_dir()
    assert not (tmp_path / "comparison" / "yolov7-tiny").exists()


def test_ultralytics_adapter_trains_and_validates_in_isolated_directories(
    tmp_path, monkeypatch
):
    calls = []

    class Box:
        map50 = 0.75
        map = 0.5
        mp = 0.8
        mr = 0.625

    class FakeModel:
        def __init__(self, weights):
            calls.append(("init", weights))
            self.trainer = types.SimpleNamespace()

        def train(self, **kwargs):
            calls.append(("train", kwargs))
            return types.SimpleNamespace(save_dir=tmp_path / "runs" / "yolov8n")

        def val(self, **kwargs):
            calls.append(("val", kwargs))
            return types.SimpleNamespace(box=Box())

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    data_yaml = tmp_path / "data.yaml"
    output_root = tmp_path / "comparisons"

    row = run_ultralytics_model(
        ModelSpec("yolov8n", "ultralytics", "yolov8n.pt"),
        data_yaml,
        output_root,
        epochs=3,
        batch=4,
        imgsz=320,
        seed=9,
        device="cpu",
    )

    assert calls[0] == ("init", "yolov8n.pt")
    assert calls[1][1] == {
        "data": str(data_yaml),
        "epochs": 3,
        "batch": 4,
        "imgsz": 320,
        "seed": 9,
        "project": str(output_root),
        "name": "yolov8n",
        "exist_ok": False,
        "device": "cpu",
    }
    assert calls[2][1] == {
        "data": str(data_yaml),
        "split": "test",
        "imgsz": 320,
        "project": str(output_root),
        "name": "yolov8n-test",
        "exist_ok": False,
        "device": "cpu",
    }
    assert row == {
        "model": "yolov8n",
        "run_dir": str(tmp_path / "runs" / "yolov8n"),
        "best_weights": str(tmp_path / "runs" / "yolov8n" / "weights" / "best.pt"),
        "map50": 0.75,
        "map": 0.5,
        "precision": 0.8,
        "recall": 0.625,
    }
    assert not config.DETECTOR_WEIGHTS.exists() or config.DETECTOR_WEIGHTS != output_root / "yolov8n" / "weights" / "best.pt"


def test_ultralytics_adapter_reports_suffixed_rerun_paths_from_trainer(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "comparisons" / "yolo11s2"
    best_weights = run_dir / "weights" / "best.pt"

    class FakeModel:
        def __init__(self, weights):
            self.trainer = types.SimpleNamespace()

        def train(self, **kwargs):
            self.trainer.save_dir = run_dir
            self.trainer.best = best_weights
            return types.SimpleNamespace()

        def val(self, **kwargs):
            return types.SimpleNamespace(
                box=types.SimpleNamespace(map50=0.0, map=0.0, mp=0.0, mr=0.0)
            )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    row = run_ultralytics_model(
        ModelSpec("yolo11s", "ultralytics", "yolo11s.pt"),
        tmp_path / "data.yaml",
        tmp_path / "comparisons",
        epochs=1,
        batch=1,
        imgsz=64,
        seed=42,
    )

    assert row["run_dir"] == str(run_dir)
    assert row["best_weights"] == str(best_weights)


def test_missing_metrics_are_serialized_as_none(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self, weights):
            pass

        def train(self, **kwargs):
            return None

        def val(self, **kwargs):
            return types.SimpleNamespace(box=types.SimpleNamespace(map50=None))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    with pytest.warns(RuntimeWarning) as captured:
        row = run_ultralytics_model(
            ModelSpec("yolo11s", "ultralytics", "yolo11s.pt"),
            tmp_path / "data.yaml",
            tmp_path / "output",
            epochs=1,
            batch=1,
            imgsz=64,
            seed=42,
        )

    assert row["map50"] is None
    assert row["map"] is None
    assert row["precision"] is None
    assert row["recall"] is None
    assert len(captured) == 4
    assert all("recording None" in str(item.message) for item in captured)


def test_invalid_ultralytics_metrics_are_none_with_warnings(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self, weights):
            pass

        def train(self, **kwargs):
            return None

        def val(self, **kwargs):
            return types.SimpleNamespace(
                box=types.SimpleNamespace(
                    map50=float("nan"),
                    map=float("inf"),
                    mp=1.01,
                    mr=-0.01,
                )
            )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    with pytest.warns(RuntimeWarning) as captured:
        row = run_ultralytics_model(
            ModelSpec("yolo11s", "ultralytics", "yolo11s.pt"),
            tmp_path / "data.yaml",
            tmp_path / "output",
            epochs=1,
            batch=1,
            imgsz=64,
            seed=42,
        )

    assert all(row[name] is None for name in ("map50", "map", "precision", "recall"))
    assert len(captured) == 4
    assert all("unavailable or invalid" in str(item.message) for item in captured)


@pytest.mark.parametrize("seed", [-1, 4_294_967_296])
def test_yolov7_rejects_seeds_outside_pythonhashseed_range(tmp_path, seed):
    with pytest.raises(ValueError, match="0 through 4294967295"):
        run_yolov7_model(
            ModelSpec("yolov7", "yolov7", "yolov7.pt"),
            tmp_path / "data.yaml",
            tmp_path / "comparison",
            epochs=1,
            batch=1,
            imgsz=64,
            seed=seed,
            yolov7_root=tmp_path / "missing-yolov7",
        )


@pytest.mark.parametrize("seed", [-1, 4_294_967_296])
def test_comparison_api_rejects_invalid_seed_before_starting_models(tmp_path, seed):
    with pytest.raises(ValueError, match="0 through 4294967295"):
        run_comparison(
            [ModelSpec("yolov8n", "ultralytics", "yolov8n.pt")],
            tmp_path / "data.yaml",
            tmp_path / "comparison",
            epochs=1,
            batch=1,
            imgsz=64,
            seed=seed,
        )


def test_run_comparison_writes_complete_json_and_csv_after_runs(tmp_path, monkeypatch):
    expected = {
        "model": "yolov8n",
        "run_dir": str(tmp_path / "summary" / "yolov8n2"),
        "best_weights": str(tmp_path / "summary" / "yolov8n2" / "weights" / "best.pt"),
        "map50": 0.75,
        "map": 0.5,
        "precision": 0.8,
        "recall": None,
    }
    monkeypatch.setattr(
        "training.yolo_comparison.run_ultralytics_model",
        lambda *args, **kwargs: expected,
    )
    rows = run_comparison(
        [ModelSpec("yolov8n", "ultralytics", "yolov8n.pt")],
        tmp_path / "data.yaml",
        tmp_path / "summary",
        epochs=1,
        batch=1,
        imgsz=64,
        seed=42,
    )

    assert rows == [expected]
    assert json.loads((tmp_path / "summary" / "summary.json").read_text()) == rows
    with (tmp_path / "summary" / "summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0] == {
        "model": "yolov8n",
        "run_dir": expected["run_dir"],
        "best_weights": expected["best_weights"],
        "map50": "0.75",
        "map": "0.5",
        "precision": "0.8",
        "recall": "",
    }


def test_comparison_does_not_write_production_weights(tmp_path, monkeypatch):
    production_weights = tmp_path / "production" / "detector.pt"
    monkeypatch.setattr(config, "DETECTOR_WEIGHTS", production_weights)
    monkeypatch.setattr(
        "training.yolo_comparison.run_ultralytics_model",
        lambda *args, **kwargs: {
            "model": args[0].name,
            "run_dir": str(tmp_path / "comparison" / args[0].name),
            "best_weights": str(tmp_path / "comparison" / args[0].name / "weights" / "best.pt"),
            "map50": None,
            "map": None,
            "precision": None,
            "recall": None,
        },
    )

    run_comparison(
        [ModelSpec("yolov8n", "ultralytics", "yolov8n.pt")],
        tmp_path / "data.yaml",
        tmp_path / "comparison",
        epochs=1,
        batch=1,
        imgsz=64,
        seed=42,
    )

    assert not production_weights.exists()


def test_train_detector_copies_best_weights_from_each_actual_run(tmp_path, monkeypatch):
    from training import train_detector

    data_root = tmp_path / "detector"
    project_root = tmp_path / "project"
    production_weights = tmp_path / "production" / "detector.pt"
    monkeypatch.setattr(train_detector.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(train_detector.config, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(train_detector.config, "DETECTOR_WEIGHTS", production_weights)
    monkeypatch.setattr(sys, "argv", ["train_detector"])

    runs = iter(("corn_detector", "corn_detector2"))

    class FakeModel:
        def __init__(self, weights):
            self.trainer = types.SimpleNamespace()

        def train(self, **kwargs):
            run_dir = project_root / "runs" / "detect" / next(runs)
            best = run_dir / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_text(run_dir.name, encoding="utf-8")
            self.trainer.save_dir = run_dir
            self.trainer.best = best
            return types.SimpleNamespace(save_dir=run_dir)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    train_detector.main()
    assert production_weights.read_text(encoding="utf-8") == "corn_detector"

    train_detector.main()
    assert production_weights.read_text(encoding="utf-8") == "corn_detector2"


def test_dry_run_does_not_import_ultralytics_in_fresh_process(tmp_path):
    blocker = tmp_path / "blocker"
    package = blocker / "ultralytics"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise AssertionError('ultralytics was imported during dry-run')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(blocker), str(tmp_path)))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.yolo_comparison",
            "--models",
            "yolov8n",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "yolov8n" in result.stdout
    assert not (tmp_path / "output" / "summary.json").exists()


def test_yolov7_numpy_shim_aliases_trapz_only_in_native_child(tmp_path):
    output_root = tmp_path / "comparison"
    shim_path = yolo_comparison._write_yolov7_numpy_shim(output_root)
    fake_numpy = tmp_path / "fake-numpy" / "numpy"
    fake_numpy.mkdir(parents=True)
    (fake_numpy / "__init__.py").write_text(
        "trapezoid = object()\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(shim_path.parent), str(tmp_path / "fake-numpy")))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import numpy; assert numpy.trapz is numpy.trapezoid",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert shim_path.read_text(encoding="utf-8").count("np.trapz = np.trapezoid") == 1


def test_yolov7_dry_run_reports_numpy_compatibility(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.yolo_comparison",
            "--models",
            "yolov7",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "numpy.trapz -> numpy.trapezoid when needed" in result.stdout
    assert "generated sitecustomize=" in result.stdout
    assert "native PYTHONPATH prepends=" in result.stdout


def _make_yolov7_checkout(tmp_path):
    root = tmp_path / "yolov7"
    (root / "cfg" / "training").mkdir(parents=True)
    (root / "train.py").write_text("", encoding="utf-8")
    (root / "test.py").write_text("", encoding="utf-8")
    for config_path in (
        "yolov7-tiny.yaml",
        "yolov7.yaml",
        "yolov7x.yaml",
    ):
        (root / "cfg" / "training" / config_path).write_text("", encoding="utf-8")
    return root


def test_yolov7_config_mapping_is_native_and_explicit():
    assert yolov7_config("yolov7-tiny") == "cfg/training/yolov7-tiny.yaml"
    assert yolov7_config("yolov7") == "cfg/training/yolov7.yaml"
    assert yolov7_config("yolov7x") == "cfg/training/yolov7x.yaml"


def test_yolov7_command_builders_include_full_native_commands(tmp_path):
    root = tmp_path / "yolov7"
    data_yaml = tmp_path / "data.yaml"
    output_root = tmp_path / "comparison"
    spec = ModelSpec("yolov7x", "yolov7", "yolov7x.pt")

    train_command = build_yolov7_train_command(
        spec,
        root,
        data_yaml,
        output_root,
        run_name="yolov7x-run-abc",
        epochs=12,
        batch=3,
        imgsz=512,
        seed=17,
        weights="/weights/yolov7x.pt",
        yolov7_python="/venvs/yolov7/bin/python",
        device="0",
    )
    assert train_command == [
        "/venvs/yolov7/bin/python",
        str(root / "train.py"),
        "--data",
        str(data_yaml.resolve()),
        "--cfg",
        str(root / "cfg/training/yolov7x.yaml"),
        "--weights",
        "/weights/yolov7x.pt",
        "--epochs",
        "12",
        "--batch-size",
        "3",
        "--img-size",
        "512",
        "--project",
        str(output_root.resolve()),
        "--name",
        "yolov7x-run-abc",
        "--device",
        "0",
    ]

    test_command = build_yolov7_test_command(
        spec,
        root,
        data_yaml,
        output_root,
        run_name="yolov7x-run-abc",
        weights="/comparison/yolov7x/weights/best.pt",
        batch=3,
        imgsz=512,
        yolov7_python="/venvs/yolov7/bin/python",
        device="0",
    )
    assert test_command == [
        "/venvs/yolov7/bin/python",
        str(root / "test.py"),
        "--data",
        str(data_yaml.resolve()),
        "--weights",
        "/comparison/yolov7x/weights/best.pt",
        "--batch-size",
        "3",
        "--img-size",
        "512",
        "--task",
        "test",
        "--project",
        str(output_root.resolve()),
        "--name",
        "yolov7x-run-abc-test",
        "--device",
        "0",
    ]


def test_yolov7_missing_root_fails_before_any_model_starts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        yolo_comparison,
        "run_ultralytics_model",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="official YOLOv7 checkout") as exc_info:
        run_comparison(
            [
                ModelSpec("yolov7", "yolov7", "yolov7.pt"),
                ModelSpec("yolov8n", "ultralytics", "yolov8n.pt"),
            ],
            tmp_path / "data.yaml",
            tmp_path / "comparison",
            epochs=1,
            batch=1,
            imgsz=64,
            seed=42,
        )

    assert "--yolov7-root" in str(exc_info.value)
    assert "No training was started" in str(exc_info.value)
    assert calls == []
    assert not (tmp_path / "comparison").exists()


def test_yolov7_native_run_uses_local_weights_preserves_output_and_parses_metrics(
    tmp_path, monkeypatch
):
    root = _make_yolov7_checkout(tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    local_weights = weights_dir / "yolov7-tiny.pt"
    local_weights.write_text("weights", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "test.py" in command[1]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="all 12 24 0.81 0.72 0.65 0.43\n",
                stderr="test warning\n",
            )
        project = Path(command[command.index("--project") + 1])
        name = command[command.index("--name") + 1]
        (project / name).mkdir(parents=True)
        return subprocess.CompletedProcess(
            command, 0, stdout="train output\n", stderr="train warning\n"
        )

    monkeypatch.setattr(yolo_comparison.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    with pytest.warns(RuntimeWarning, match="Official YOLOv7 cannot use MPS"):
        row = run_yolov7_model(
            ModelSpec("yolov7-tiny", "yolov7", "yolov7-tiny.pt"),
            tmp_path / "data.yaml",
            "comparison",
            epochs=2,
            batch=4,
            imgsz=320,
            seed=9,
            yolov7_root=root,
            yolov7_python=sys.executable,
            yolov7_weights_dir=weights_dir,
            device="mps",
        )

    assert len(calls) == 2
    train_command = calls[0][0]
    assert train_command[1] == str(root / "train.py")
    assert "--seed" not in train_command
    assert train_command[train_command.index("--weights") + 1] == str(
        local_weights.resolve()
    )
    assert train_command[train_command.index("--project") + 1] == str(
        (tmp_path / "comparison").resolve()
    )
    assert train_command[train_command.index("--device") + 1] == "cpu"
    test_command = calls[1][0]
    assert test_command[test_command.index("--device") + 1] == "cpu"
    assert calls[0][1]["cwd"] == str(root.resolve())
    assert calls[0][1]["env"]["PYTHONHASHSEED"] == "9"
    assert calls[0][1]["env"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert calls[1][1]["env"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    shim_dir = str((tmp_path / "comparison" / "generated").resolve())
    assert calls[0][1]["env"]["PYTHONPATH"].split(os.pathsep)[0] == shim_dir
    assert calls[1][1]["env"]["PYTHONPATH"] == calls[0][1]["env"]["PYTHONPATH"]
    run_dir = tmp_path / "comparison" / Path(row["run_dir"]).name
    metadata = json.loads((run_dir / "run-metadata.json").read_text())
    assert metadata["seed"] == 9
    assert metadata["environment"]["PYTHONHASHSEED"] == "9"
    assert metadata["environment"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert metadata["compatibility"]["sitecustomize"] == str(
        (tmp_path / "comparison" / "generated" / "sitecustomize.py").resolve()
    )
    assert "numpy.trapz" in metadata["compatibility"]["behavior"]
    assert metadata["requested_device"] == "mps"
    assert metadata["effective_yolov7_device"] == "cpu"
    assert "--seed" not in metadata["train_command"]
    assert (run_dir / "train-stdout.txt").read_text() == "train output\n"
    assert (run_dir / "train-stderr.txt").read_text() == "train warning\n"
    assert row["map50"] == 0.65
    assert row["map"] == 0.43
    assert row["precision"] == 0.81
    assert row["recall"] == 0.72


def test_run_comparison_passes_generated_native_yaml_to_both_yolov7_commands(
    tmp_path, monkeypatch
):
    root = _make_yolov7_checkout(tmp_path)
    data_root = tmp_path / "detector"
    output_root = tmp_path / "comparison"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "train.py" in command[1]:
            project = Path(command[command.index("--project") + 1])
            name = command[command.index("--name") + 1]
            (project / name / "weights").mkdir(parents=True)
            (project / name / "weights" / "best.pt").write_text("best")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command, 0, stdout="all 1 1 0.8 0.7 0.6 0.5\n", stderr=""
        )

    monkeypatch.setattr(yolo_comparison.subprocess, "run", fake_run)
    with pytest.warns(RuntimeWarning, match="does not accept --seed"):
        run_comparison(
            [ModelSpec("yolov7", "yolov7", "yolov7.pt")],
            data_root / "data.yaml",
            output_root,
            epochs=1,
            batch=1,
            imgsz=64,
            seed=42,
            yolov7_root=root,
        )

    native_yaml = output_root / "generated" / "yolov7-data.yaml"
    assert native_yaml.is_file()
    assert len(calls) == 2
    for command, _ in calls:
        assert command[command.index("--data") + 1] == str(native_yaml.resolve())
    assert native_yaml.read_text(encoding="utf-8").startswith(
        f"train: {(data_root / 'images/train').resolve().as_posix()}\n"
    )
    run_dir = next(path for path in output_root.iterdir() if path.name.startswith("yolov7-"))
    metadata = json.loads((run_dir / "run-metadata.json").read_text())
    assert metadata["data_yaml"] == str(native_yaml.resolve())


def test_yolov7_native_run_ignores_stale_and_concurrent_suffixes(tmp_path, monkeypatch):
    root = _make_yolov7_checkout(tmp_path)
    output_root = tmp_path / "comparison"
    intended_run_dir = output_root / "yolov7-tiny"
    stale_run_dir = output_root / "yolov7-tiny2"
    higher_stale_run_dir = output_root / "yolov7-tiny17"
    for stale_dir in (intended_run_dir, stale_run_dir, higher_stale_run_dir):
        (stale_dir / "weights").mkdir(parents=True)
        (stale_dir / "weights" / "best.pt").write_text(
            f"stale {stale_dir.name}", encoding="utf-8"
        )
    monkeypatch.setattr(
        yolo_comparison,
        "_unique_run_dir",
        lambda output_root, spec: intended_run_dir,
    )
    (root / "train.py").write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--project', required=True)
parser.add_argument('--name', required=True)
args, _ = parser.parse_known_args()
run_dir = Path(args.project) / f'{args.name}3'
(run_dir / 'weights').mkdir(parents=True)
(run_dir / 'weights' / 'best.pt').write_text('native best', encoding='utf-8')
print('native train output')
""",
        encoding="utf-8",
    )
    (root / "test.py").write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True)
args, _ = parser.parse_known_args()
if not Path(args.weights).is_file():
    raise SystemExit(f'missing checkpoint: {args.weights}')
print('all 12 24 0.91 0.82 0.73 0.64')
""",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="does not accept --seed"):
        row = run_yolov7_model(
            ModelSpec("yolov7-tiny", "yolov7", "yolov7-tiny.pt"),
            tmp_path / "data.yaml",
            output_root,
            epochs=1,
            batch=1,
            imgsz=64,
            seed=42,
            yolov7_root=root,
            yolov7_python=sys.executable,
        )

    run_dir = Path(row["run_dir"])
    assert run_dir == output_root / "yolov7-tiny3"
    assert run_dir.parent == output_root
    best_weights = run_dir / "weights" / "best.pt"
    assert row["best_weights"] == str(best_weights)
    assert best_weights.read_text(encoding="utf-8") == "native best"
    metadata = json.loads((run_dir / "run-metadata.json").read_text())
    assert metadata["test_command"][metadata["test_command"].index("--weights") + 1] == str(
        best_weights
    )
    assert (run_dir / "train-stdout.txt").read_text() == "native train output\n"
    assert (run_dir / "test-stdout.txt").read_text() == (
        "all 12 24 0.91 0.82 0.73 0.64\n"
    )
    assert row["map50"] == 0.73
    assert row["map"] == 0.64
    assert row["precision"] == 0.91
    assert row["recall"] == 0.82


def test_yolov7_native_run_rejects_missing_new_run_instead_of_using_stale_output(
    tmp_path, monkeypatch
):
    root = _make_yolov7_checkout(tmp_path)
    output_root = tmp_path / "comparison"
    intended_run_dir = output_root / "yolov7"
    (intended_run_dir / "weights").mkdir(parents=True)
    (intended_run_dir / "weights" / "best.pt").write_text(
        "stale best", encoding="utf-8"
    )
    monkeypatch.setattr(
        yolo_comparison,
        "_unique_run_dir",
        lambda output_root, spec: intended_run_dir,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(yolo_comparison.subprocess, "run", fake_run)
    with pytest.warns(RuntimeWarning, match="does not accept --seed"):
        with pytest.raises(
            RuntimeError, match="no new matching run directory was created"
        ):
            run_yolov7_model(
                ModelSpec("yolov7", "yolov7", "yolov7.pt"),
                tmp_path / "data.yaml",
                output_root,
                epochs=1,
                batch=1,
                imgsz=64,
                seed=42,
                yolov7_root=root,
            )

    assert len(calls) == 1
    assert calls[0][0][1] == str(root / "train.py")
    assert not (intended_run_dir / "run-metadata.json").exists()


def test_yolov7_unrecognized_metrics_are_none_with_warning(tmp_path, monkeypatch):
    root = _make_yolov7_checkout(tmp_path)

    def fake_run(command, **kwargs):
        if "train.py" in command[1]:
            project = Path(command[command.index("--project") + 1])
            name = command[command.index("--name") + 1]
            (project / name).mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="new format", stderr="")

    monkeypatch.setattr(yolo_comparison.subprocess, "run", fake_run)
    with pytest.warns(RuntimeWarning) as captured:
        row = run_yolov7_model(
            ModelSpec("yolov7", "yolov7", "yolov7.pt"),
            tmp_path / "data.yaml",
            tmp_path / "comparison",
            epochs=1,
            batch=1,
            imgsz=64,
            seed=42,
            yolov7_root=root,
        )

    messages = [str(item.message) for item in captured]
    assert any("does not accept --seed" in message for message in messages)
    assert any("Could not parse" in message for message in messages)
    assert row["map50"] is None
    assert row["map"] is None
    assert row["precision"] is None
    assert row["recall"] is None


def test_yolov7_train_failure_is_actionable_and_preserves_rejection_output(
    tmp_path, monkeypatch
):
    root = _make_yolov7_checkout(tmp_path)

    def fake_run(command, **kwargs):
        project = Path(command[command.index("--project") + 1])
        name = command[command.index("--name") + 1]
        (project / name).mkdir(parents=True)
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="native train failed\n",
        )

    monkeypatch.setattr(yolo_comparison.subprocess, "run", fake_run)
    with pytest.warns(RuntimeWarning, match="does not accept --seed"):
        with pytest.raises(RuntimeError, match="PYTHONHASHSEED") as exc_info:
            run_yolov7_model(
                ModelSpec("yolov7", "yolov7", "yolov7.pt"),
                tmp_path / "data.yaml",
                tmp_path / "comparison",
                epochs=1,
                batch=1,
                imgsz=64,
                seed=42,
                yolov7_root=root,
            )

    assert "yolov7-" in str(exc_info.value)
    run_dir = next(
        path
        for path in (tmp_path / "comparison").iterdir()
        if path.name.startswith("yolov7-")
    )
    assert (run_dir / "train-stderr.txt").read_text() == (
        "native train failed\n"
    )
    metadata = json.loads((run_dir / "run-metadata.json").read_text())
    assert metadata["seed"] == 42
    assert "--seed" not in metadata["train_command"]


@pytest.mark.parametrize(
    "output",
    [
        "all 12 24 0.81 0.72 0.65 0.43 extra",
        "all 12 24 0.81 0.72 0.65",
        "all twelve 24 0.81 0.72 0.65 0.43",
        "all 12 24 nan 0.72 0.65 0.43",
        "all 12 24 0.81 inf 0.65 0.43",
        "all 12 24 0.81 0.72 -inf 0.43",
    ],
)
def test_yolov7_metrics_parser_rejects_malformed_and_nonfinite_rows(output):
    assert _parse_yolov7_metrics(output) is None


def test_run_comparison_resolves_relative_output_root_for_summary(tmp_path, monkeypatch):
    root = _make_yolov7_checkout(tmp_path)
    monkeypatch.chdir(tmp_path)
    observed = []

    def fake_run_model(spec, data_yaml, output_root, **kwargs):
        observed.append(output_root)
        return {
            "model": spec.name,
            "run_dir": str(output_root / "run"),
            "best_weights": str(output_root / "run" / "weights" / "best.pt"),
            "map50": None,
            "map": None,
            "precision": None,
            "recall": None,
        }

    monkeypatch.setattr(yolo_comparison, "_run_model", fake_run_model)
    run_comparison(
        [ModelSpec("yolov7", "yolov7", "yolov7.pt")],
        tmp_path / "data.yaml",
        "relative-comparison",
        epochs=1,
        batch=1,
        imgsz=64,
        seed=42,
        yolov7_root=root,
    )

    expected_root = (tmp_path / "relative-comparison").resolve()
    assert observed == [expected_root]
    assert (expected_root / "summary.json").exists()
    assert (expected_root / "summary.csv").exists()


@pytest.mark.parametrize("output_suffix", ["", "nested"])
def test_run_comparison_rejects_output_root_in_production_weights(
    tmp_path, monkeypatch, output_suffix
):
    weights_root = tmp_path / "data" / "weights"
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_root)
    calls = []
    monkeypatch.setattr(
        yolo_comparison,
        "_run_model",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    output_root = weights_root / output_suffix

    with pytest.raises(ValueError, match="must not equal or be inside"):
        run_comparison(
            [ModelSpec("yolov8n", "ultralytics", "yolov8n.pt")],
            tmp_path / "data.yaml",
            output_root,
            epochs=1,
            batch=1,
            imgsz=64,
            seed=42,
        )

    assert calls == []
    assert not output_root.exists()


def test_dry_run_prints_complete_plans_for_all_seven_models(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.yolo_comparison",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
            "--yolov7-root",
            str(tmp_path / "yolov7"),
            "--dry-run",
        ],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for name in model_names():
        assert name in result.stdout
    assert result.stdout.count("train.py") == 3
    assert "test.py" in result.stdout
    assert "PYTHONHASHSEED=42" in result.stdout
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in result.stdout
    assert "trusted official YOLOv7 checkpoints only" in result.stdout
    assert "YOLO('yolov8n.pt')" in result.stdout
    assert not (tmp_path / "output" / "summary.json").exists()


def test_ultralytics_dry_run_includes_device_for_train_and_validation(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.yolo_comparison",
            "--models",
            "yolov8n",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
            "--device",
            "cuda:0",
            "--dry-run",
        ],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("device='cuda:0'") == 2


def test_mps_dry_run_maps_only_official_yolov7_to_cpu(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.yolo_comparison",
            "--models",
            "yolov7,yolov8n",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
            "--yolov7-root",
            str(tmp_path / "yolov7"),
            "--device",
            "mps",
            "--dry-run",
        ],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--device cpu") == 2
    assert "requested device='mps'" in result.stdout
    assert "effective YOLOv7 device='cpu'" in result.stdout
    assert "WARNING: official YOLOv7 cannot use MPS here; running on CPU" in result.stdout
    assert result.stdout.count(", device='mps'") == 2


def test_cli_rejects_invalid_seed_before_dry_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.yolo_comparison",
            "--models",
            "yolov8n",
            "--seed",
            "-1",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "seed must be from 0 through 4294967295" in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "output").exists()
