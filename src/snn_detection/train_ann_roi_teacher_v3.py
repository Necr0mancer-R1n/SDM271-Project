import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "cnn_detection"))

from voc_dataset import VOC_CLASSES

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "snn_detection_v3"
ROI_DIR = OUTPUT_ROOT / "roi_dataset"
MODEL_DIR = OUTPUT_ROOT / "models"
RESULT_DIR = OUTPUT_ROOT / "results"
FIGURE_DIR = OUTPUT_ROOT / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 21
CROP_SIZE = 64
PADDING_RATIO = 0.15


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RoiCsvDataset(Dataset):
    def __init__(self, csv_path, crop_size=64, padding_ratio=0.15, augment=False):
        self.csv_path = Path(csv_path)
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        self.augment = augment
        if not self.csv_path.exists():
            raise FileNotFoundError(f"ROI CSV not found: {self.csv_path}")
        self.rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["label"] = int(row["label"])
                row["box"] = json.loads(row["box"])
                row["matched_iou"] = float(row["matched_iou"])
                row["proposal_score"] = float(row["proposal_score"])
                self.rows.append(row)
        print(f"Loaded {len(self.rows)} samples from {self.csv_path}")

    def __len__(self):
        return len(self.rows)

    def crop_roi(self, image, box):
        width, height = image.size
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        pad_x, pad_y = box_w * self.padding_ratio, box_h * self.padding_ratio
        x1 = max(0, int(x1 - pad_x)); y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x)); y2 = min(height, int(y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height
        return image.crop((x1, y1, x2, y2)).resize((self.crop_size, self.crop_size))

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        crop = self.crop_roi(image, row["box"])
        if self.augment:
            if random.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.25:
                arr = np.asarray(crop).astype(np.float32)
                factor = random.uniform(0.95, 1.05)
                arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
                crop = Image.fromarray(arr)
        return F.to_tensor(crop), torch.tensor(row["label"], dtype=torch.long)


class AnnRoiTeacherV3(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def compute_class_weights(dataset, num_classes=21, max_weight=3.0):
    labels = [row["label"] for row in dataset.rows]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.25, max_weight)
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int).tolist()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
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
    total_loss = total_correct = total_samples = 0
    non_bg_correct = non_bg_total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        non_bg = labels != 0
        non_bg_correct += ((preds == labels) & non_bg).sum().item()
        non_bg_total += non_bg.sum().item()
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "non_background_accuracy": non_bg_correct / max(non_bg_total, 1),
    }


def save_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("ANN ROI Teacher v3 Loss")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ann_teacher_v3_loss_curve.png", dpi=200); plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.plot(epochs, history["val_non_bg_acc"], label="Val Non-bg Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("ANN ROI Teacher v3 Accuracy")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ann_teacher_v3_accuracy_curve.png", dpi=200); plt.close()


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataset = RoiCsvDataset(ROI_DIR / "train.csv", crop_size=CROP_SIZE, padding_ratio=PADDING_RATIO, augment=True)
    val_dataset = RoiCsvDataset(ROI_DIR / "val.csv", crop_size=CROP_SIZE, padding_ratio=PADDING_RATIO, augment=False)
    class_weights, class_counts = compute_class_weights(train_dataset, num_classes=NUM_CLASSES, max_weight=3.0)
    print("Class counts:", class_counts)
    print("Class weights:", [round(float(x), 4) for x in class_weights.tolist()])
    batch_size, num_epochs = 128, 15
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = AnnRoiTeacherV3(num_classes=NUM_CLASSES).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = 0.0
    best_model_path = MODEL_DIR / "ann_roi_teacher_v3_best.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_non_bg_acc": []}
    import time
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"]); history["val_acc"].append(val_metrics["accuracy"])
        history["val_non_bg_acc"].append(val_metrics["non_background_accuracy"])
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"model_state_dict": model.state_dict(), "model_name": "AnnRoiTeacherV3", "num_classes": NUM_CLASSES, "class_names": VOC_CLASSES, "crop_size": CROP_SIZE, "padding_ratio": PADDING_RATIO, "best_validation_accuracy": best_val_acc, "history": history}, best_model_path)
        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} Val Non-bg Acc: {val_metrics['non_background_accuracy']:.4f} Time: {time.time() - epoch_start:.2f}s")
    total_time = time.time() - start_time
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate(model, val_loader, criterion, device)
    metrics = {"model": "AnnRoiTeacherV3", "dataset": "SNN detection v3 ROI dataset", "num_classes": NUM_CLASSES, "class_names": VOC_CLASSES, "crop_size": CROP_SIZE, "padding_ratio": PADDING_RATIO, "epochs": num_epochs, "batch_size": batch_size, "optimizer": "AdamW", "learning_rate": 5e-4, "weight_decay": 1e-4, "class_counts": class_counts, "class_weights": [float(x) for x in class_weights.tolist()], "best_validation_accuracy": best_val_acc, "final_validation_loss": final_metrics["loss"], "final_validation_accuracy": final_metrics["accuracy"], "final_non_background_accuracy": final_metrics["non_background_accuracy"], "total_training_time_seconds": total_time, "history": history}
    with open(RESULT_DIR / "ann_teacher_v3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    save_curves(history)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final validation loss: {final_metrics['loss']:.4f}")
    print(f"Final validation accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final non-background accuracy: {final_metrics['non_background_accuracy']:.4f}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {RESULT_DIR / 'ann_teacher_v3_metrics.json'}")


if __name__ == "__main__":
    main()
