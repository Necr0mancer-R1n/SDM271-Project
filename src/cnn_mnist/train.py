import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cnn_mnist"
MODEL_DIR = OUTPUT_ROOT / "models"
RESULT_DIR = OUTPUT_ROOT / "results"
FIGURE_DIR = OUTPUT_ROOT / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CNNMnist(nn.Module):
    """
    输入: 1 x 28 x 28
    Conv1: 1 -> 32, kernel=3, padding=1
    MaxPool: 28 -> 14
    Conv2: 32 -> 64, kernel=3, padding=1
    MaxPool: 14 -> 7
    FC: 64*7*7 -> 128 -> 10
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def save_prediction_examples(model, loader, device, save_path, num_examples=12):
    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    preds = logits.argmax(dim=1)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    n = min(num_examples, images.size(0))

    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        plt.title(f"Label: {labels[i].item()} | Pred: {preds[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_loss_curve(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN-MNIST Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_val_dataset = datasets.MNIST(
        root=str(DATA_ROOT),
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=str(DATA_ROOT),
        train=False,
        transform=transform,
        download=True
    )

    # MNIST 官方训练集 60000 张，拆成 50000 训练 + 10000 验证；
    # 官方测试集 10000 张作为测试集，整体约等于 5:1:1。
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [50000, 10000],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 128

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = CNNMnist().to(device)
    print(model)
    print(f"Trainable parameters: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 8

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_path = MODEL_DIR / "cnn_mnist_best.pt"

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "CNNMnist",
                "val_acc": best_val_acc,
                "history": history
            }, best_model_path)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - start_time

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Total training time: {total_time:.2f}s")

    metrics = {
        "model": "CNNMnist",
        "dataset": "MNIST",
        "split": {
            "train": 50000,
            "validation": 10000,
            "test": 10000
        },
        "batch_size": batch_size,
        "epochs": num_epochs,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "criterion": "CrossEntropyLoss",
        "best_validation_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "trainable_parameters": count_parameters(model),
        "total_training_time_seconds": total_time,
        "device": str(device),
        "history": history
    }

    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    save_loss_curve(history, FIGURE_DIR / "loss_curve.png")
    save_prediction_examples(
        model,
        test_loader,
        device,
        FIGURE_DIR / "sample_predictions.png"
    )

    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {RESULT_DIR / 'metrics.json'}")
    print(f"Saved loss curve to: {FIGURE_DIR / 'loss_curve.png'}")
    print(f"Saved sample predictions to: {FIGURE_DIR / 'sample_predictions.png'}")


if __name__ == "__main__":
    main()
