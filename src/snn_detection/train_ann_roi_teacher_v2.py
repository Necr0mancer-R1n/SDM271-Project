"""
Train an ANN ROI classifier as teacher for SNN target detection.

Input dataset:
    outputs/snn_detection_v2/proposal_rois/train.csv
    outputs/snn_detection_v2/proposal_rois/val.csv

Output:
    outputs/snn_detection_v2/models/ann_roi_teacher_v2_best.pt
    outputs/snn_detection_v2/results/ann_teacher_v2_metrics.json
"""

import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "cnn_detection"))

from voc_dataset import VOC_CLASSES  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "snn_detection_v2"
ROI_DIR = OUTPUT_ROOT / "proposal_rois"
MODEL_DIR = OUTPUT_ROOT / "models"
RESULT_DIR = OUTPUT_ROOT / "results"
FIGURE_DIR = OUTPUT_ROOT / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

VOC20_CLASSES = VOC_CLASSES[1:]

# Training settings
SEED = 42
CROP_SIZE = 64
BATCH_SIZE = 128
NUM_EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.03
USE_CLASS_WEIGHTS = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProposalROIDataset(Dataset):
    def __init__(self, csv_path: Path, crop_size=64, augment=False):
        self.csv_path = Path(csv_path)
        self.crop_size = crop_size
        self.augment = augment

        if not self.csv_path.exists():
            raise FileNotFoundError(f"ROI CSV not found: {self.csv_path}")

        self.samples = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "image_path": Path(row["image_path"]),
                    "box": [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])],
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                    "source": row["source"],
                    "matched_iou": float(row["matched_iou"]),
                })

        print(f"Loaded {len(self.samples)} ROI samples from {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def crop_roi(self, image, box):
        width, height = image.size
        x1, y1, x2, y2 = box

        # Mild jitter only for ANN teacher training, not as aggressive as v2.
        if self.augment and random.random() < 0.40:
            w = x2 - x1
            h = y2 - y1
            shift_x = random.uniform(-0.06, 0.06) * w
            shift_y = random.uniform(-0.06, 0.06) * h
            scale = random.uniform(0.95, 1.08)
            cx = (x1 + x2) / 2 + shift_x
            cy = (y1 + y2) / 2 + shift_y
            nw = w * scale
            nh = h * scale
            x1 = cx - nw / 2
            x2 = cx + nw / 2
            y1 = cy - nh / 2
            y2 = cy + nh / 2

        pad_x = 0.03 * (x2 - x1)
        pad_y = 0.03 * (y2 - y1)
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))

        crop = image.crop((x1, y1, x2, y2)).resize((self.crop_size, self.crop_size))
        return crop

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        crop = self.crop_roi(image, sample["box"])

        if self.augment:
            if random.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.25:
                crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.85, 1.15))
            if random.random() < 0.25:
                crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.85, 1.15))

        tensor = TF.to_tensor(crop)  # keep [0, 1] for later ANN-to-SNN compatibility
        label = torch.tensor(sample["label"], dtype=torch.long)
        return tensor, label


class ANNROIClassifier(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(p=0.20)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_class_weights(dataset):
    labels = [sample["label"] for sample in dataset.samples]
    counts = np.bincount(labels, minlength=20).astype(np.float32)
    # Mild inverse-sqrt weighting, normalized to mean=1. More stable than full inverse frequency.
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    print("Class counts:", counts.astype(int).tolist())
    print("Class weights:", [round(float(x), 4) for x in weights.tolist()])
    return torch.tensor(weights, dtype=torch.float32)


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


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
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ANN ROI Teacher V2 Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ann_teacher_v2_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ANN ROI Teacher V2 Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ann_teacher_v2_accuracy_curve.png", dpi=200)
    plt.close()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = ROI_DIR / "train.csv"
    val_csv = ROI_DIR / "val.csv"

    train_dataset = ProposalROIDataset(train_csv, crop_size=CROP_SIZE, augment=True)
    val_dataset = ProposalROIDataset(val_csv, crop_size=CROP_SIZE, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = ANNROIClassifier(num_classes=20).to(device)
    print(model)
    print(f"Trainable parameters: {count_parameters(model)}")

    class_weights = build_class_weights(train_dataset).to(device) if USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = MODEL_DIR / "ann_roi_teacher_v2_best.pt"

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "ANNROIClassifierV2",
                "num_classes": 20,
                "voc20_classes": VOC20_CLASSES,
                "crop_size": CROP_SIZE,
                "best_val_acc": best_val_acc,
                "history": history,
            }, best_model_path)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

    total_time = time.time() - start_time

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    metrics = {
        "model": "ANN ROI Teacher V2",
        "dataset": "proposal-matched Pascal VOC ROI crops",
        "crop_size": CROP_SIZE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "best_validation_accuracy": best_val_acc,
        "final_validation_loss": final_val_loss,
        "final_validation_accuracy": final_val_acc,
        "trainable_parameters": count_parameters(model),
        "total_training_time_seconds": total_time,
        "device": str(device),
        "history": history,
    }

    with open(RESULT_DIR / "ann_teacher_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    save_curves(history)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {RESULT_DIR / 'ann_teacher_v2_metrics.json'}")


if __name__ == "__main__":
    main()
