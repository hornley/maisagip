import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from backend.app import config


def make_loader(root, target_size, train=False):
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
        batch_size=32,
        shuffle=train,
        num_workers=2,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNetV2-S for corn variety classification.")
    parser.add_argument("--data-root", default=str(config.DATA_DIR / "classifier"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-to", default=str(config.CLASSIFIER_WEIGHTS))
    args = parser.parse_args()

    train_root = Path(args.data_root) / "train"
    val_root = Path(args.data_root) / "val"
    train_loader = make_loader(train_root, config.IMAGE_TARGET_SIZE, train=True)
    val_loader = make_loader(val_root, config.IMAGE_TARGET_SIZE, train=False)

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

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * inputs.size(0)
        train_loss = running / len(train_loader.dataset)

        acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
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
            preds = model(inputs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    main()