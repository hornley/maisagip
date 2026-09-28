from zipfile import ZipFile

from training.prepare_cloud_bundle import _REQUIRED_FILES, create_bundle


def test_create_bundle_contains_portable_dataset_and_colab_instructions(tmp_path):
    project = tmp_path / "project"
    for relative in _REQUIRED_FILES:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    dataset = project / "data" / "detector"
    dataset.mkdir(parents=True)
    (dataset / "data.yaml").write_text(
        "path: /local/project/data/detector\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: [corn_ear]\n",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / f"ear001_{split}.jpg").write_bytes(b"image")
        (label_dir / f"ear001_{split}.txt").write_text("0 0.5 0.5 1 1\n")

    (dataset / "images" / "train" / "ignored.heic").write_bytes(b"heic")
    (dataset / "labels" / "train" / "ignored.json").write_text("{}")
    (project / "data" / "comparisons" / "yolo" / "local.pt").parent.mkdir(
        parents=True
    )
    (project / "data" / "comparisons" / "yolo" / "local.pt").write_bytes(b"pt")

    archive_path = create_bundle(project, tmp_path / "maisagip-cloud-training.zip")

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert all(relative.as_posix() in names for relative in _REQUIRED_FILES)
        assert "data/detector/data.yaml" in names
        assert "requirements-cloud.txt" in names
        assert "CLOUD_README.md" in names
        assert "data/detector/images/train/ignored.heic" not in names
        assert "data/detector/labels/train/ignored.json" not in names
        assert "data/comparisons/yolo/local.pt" not in names

        yaml_text = archive.read("data/detector/data.yaml").decode("utf-8")
        assert "path: data/detector" in yaml_text
        assert "train: images/train" in yaml_text
        assert "val: images/val" in yaml_text
        assert "test: images/test" in yaml_text

        readme = archive.read("CLOUD_README.md").decode("utf-8")
        assert "22 ears / 88 images" in readme
        assert "4 ears / 16 images" in readme
        assert "5 ears / 20 images" in readme
        assert "ear012, ear014" in readme
        assert "ear013" in readme
        assert "ear015" in readme
        assert "--models yolov8n,yolov8s,yolov8m,yolo11s" in readme
        assert "--seed 42" in readme
