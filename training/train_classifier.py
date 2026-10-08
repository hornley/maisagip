import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from backend.app import config


def validate_classification_split(root, expected_classes=None):
    """Validate the ImageFolder class mapping and contents before model setup."""
    root = Path(root)
    expected_classes = list(expected_classes or config.VARIETY_CLASSES)
    if expected_classes != ["white_corn", "yellow_sweet_corn"]:
        raise ValueError(
            "classifier config must contain exactly "
            "['white_corn', 'yellow_sweet_corn'] in that order"
        )
    if not root.is_dir():
        raise ValueError(f"classifier split does not exist: {root}")

    class_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    if class_dirs != expected_classes:
        raise ValueError(
            f"{root} must contain exactly {expected_classes} class directories "
            f"in config order; found {class_dirs}"
        )

    try:
        dataset = datasets.ImageFolder(root)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError(f"{root} has no valid images for every configured class") from exc
    if dataset.classes != expected_classes or dataset.class_to_idx != {
        name: index for index, name in enumerate(expected_classes)
    }:
        raise ValueError(
            f"{root} class mapping {dataset.class_to_idx} does not match "
            f"the configured mapping"
        )
    if not dataset.samples:
        raise ValueError(f"classifier split is empty: {root}")
    empty_classes = [name for name, index in dataset.class_to_idx.items()
                     if not any(sample_target == index for _, sample_target in dataset.samples)]
    if empty_classes:
        raise ValueError(f"{root} has no valid images for classes: {empty_classes}")
    return dataset


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def make_loader(root, target_size, train=False, batch_size=4):
    common = [
        transforms.Resize(int(target_size * 1.1)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ]
    if train:
        normalize = [transforms.RandomHorizontalFlip(p=0.5)]
    else:
        normalize = []
    normalize += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    transform = transforms.Compose(common + normalize)
    return DataLoader(
        datasets.ImageFolder(root, transform=transform),
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
    )


def should_save_checkpoint(accuracy, best_accuracy):
    return best_accuracy is None or accuracy > best_accuracy


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNetV2-S for corn variety classification.")
    parser.add_argument("--data-root", default=str(config.DATA_DIR / "classifier"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--save-to", default=str(config.CLASSIFIER_WEIGHTS))
    return parser


def main():
    args = build_parser().parse_args()

    train_root = Path(args.data_root) / "train"
    val_root = Path(args.data_root) / "val"
    validate_classification_split(train_root)
    validate_classification_split(val_root)
    train_loader = make_loader(
        train_root, config.IMAGE_TARGET_SIZE, train=True, batch_size=args.batch_size
    )
    val_loader = make_loader(
        val_root, config.IMAGE_TARGET_SIZE, train=False, batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(config.VARIETY_CLASSES)

    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, n_classes),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = None
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext():
                loss = criterion(model(inputs), targets)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running += loss.item() * inputs.size(0)
        train_loss = running / len(train_loader.dataset)

        acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} val_acc={acc:.4f}")

        if should_save_checkpoint(acc, best_acc):
            best_acc = acc
            weights_path = Path(args.save_to)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            print(f"saved best weights to {weights_path} (val_acc={acc:.4f})")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext():
                preds = model(inputs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    main()
