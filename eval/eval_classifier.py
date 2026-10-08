import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from backend.app import config


def build_loader(root):
    transform = transforms.Compose(
        [
            transforms.Resize(int(config.IMAGE_TARGET_SIZE * 1.1)),
            transforms.CenterCrop(config.IMAGE_TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return DataLoader(datasets.ImageFolder(root, transform=transform), batch_size=32, num_workers=2)


def validate_test_class_mapping(root, expected_classes=None):
    expected_classes = list(expected_classes or config.VARIETY_CLASSES)
    try:
        dataset = datasets.ImageFolder(root)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError(f"test dataset has no valid configured class mapping: {root}") from exc
    expected_mapping = {name: index for index, name in enumerate(expected_classes)}
    if dataset.class_to_idx != expected_mapping:
        raise ValueError(
            f"test dataset class mapping {dataset.class_to_idx} does not match "
            f"the configured mapping {expected_mapping}"
        )
    if not dataset.samples:
        raise ValueError(f"test dataset is empty: {root}")
    return dataset.class_to_idx


def classification_metrics(matrix):
    """Return accuracy and unweighted per-class (macro) metrics."""
    n = len(matrix)
    per_class = []
    for i in range(n):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((precision, recall, f1))
    total = sum(sum(row) for row in matrix)
    accuracy = sum(matrix[i][i] for i in range(n)) / total if total else 0.0
    return (
        accuracy,
        sum(item[0] for item in per_class) / n,
        sum(item[1] for item in per_class) / n,
        sum(item[2] for item in per_class) / n,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the corn variety classifier.")
    parser.add_argument("--weights", default=str(config.CLASSIFIER_WEIGHTS))
    parser.add_argument("--data-root", default=str(config.DATA_DIR / "classifier" / "test"))
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"weights not found: {args.weights}")

    validate_test_class_mapping(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_v2_s(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(1280, len(config.VARIETY_CLASSES)),
    )
    state = torch.load(args.weights, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(device)
    model.eval()

    loader = build_loader(args.data_root)
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            preds = model(inputs.to(device)).argmax(dim=1).cpu().tolist()
            y_true.extend(targets.tolist())
            y_pred.extend(preds)

    n = len(config.VARIETY_CLASSES)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    print("Class | Precision | Recall | F1 | Support")
    for i, name in enumerate(config.VARIETY_CLASSES):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        support = sum(matrix[i])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{name:14s} {precision:.4f}   {recall:.4f}  {f1:.4f}  {support}")

    accuracy, macro_precision, macro_recall, macro_f1 = classification_metrics(matrix)
    print(
        f"accuracy={accuracy:.4f} macro_precision={macro_precision:.4f} "
        f"macro_recall={macro_recall:.4f} macro_f1={macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
