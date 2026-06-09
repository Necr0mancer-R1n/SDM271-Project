"""
Train an ANN-initialized SNN ROI classifier with teacher distillation.

Prerequisites:
    1. outputs/snn_detection_v2/proposal_rois/train.csv and val.csv
    2. outputs/snn_detection_v2/models/ann_roi_teacher_v2_best.pt

Outputs:
    outputs/snn_detection_v2/models/snn_roi_distill_v2_best.pt
    outputs/snn_detection_v2/results/snn_distill_v2_metrics.json
    outputs/snn_detection_v2/figures/snn_distill_* curves and spike count figures
"""

import csv
import json
import random
import sys
import time
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

import snntorch as snn
from snntorch import spikegen


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
BATCH_SIZE = 64
NUM_EPOCHS = 16
NUM_STEPS = 25
BETA = 0.95
LR = 5e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.03
DISTILL_ALPHA = 0.50
DISTILL_TEMPERATURE = 2.0


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

        if self.augment and random.random() < 0.25:
            w = x2 - x1
            h = y2 - y1
            shift_x = random.uniform(-0.04, 0.04) * w
            shift_y = random.uniform(-0.04, 0.04) * h
            scale = random.uniform(0.97, 1.06)
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
            if random.random() < 0.15:
                crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.9, 1.1))
            if random.random() < 0.15:
                crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.9, 1.1))

        tensor = TF.to_tensor(crop)
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


class SNNROIClassifierV2(nn.Module):
    def __init__(self, num_classes=20, beta=0.95):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(beta=beta)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.lif4 = snn.Leaky(beta=beta)
        self.dropout = nn.Dropout(p=0.10)

        self.fc2 = nn.Linear(512, num_classes)
        self.lif5 = snn.Leaky(beta=beta)

    def forward(self, spike_data, return_spikes=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        output_spikes = []
        num_steps = spike_data.size(0)

        for step in range(num_steps):
            x = spike_data[step]

            cur1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            x = self.pool(spk1)

            cur2 = self.bn2(self.conv2(x))
            spk2, mem2 = self.lif2(cur2, mem2)
            x = self.pool(spk2)

            cur3 = self.bn3(self.conv3(x))
            spk3, mem3 = self.lif3(cur3, mem3)
            x = self.pool(spk3)

            x = self.gap(x)
            x = x.flatten(start_dim=1)

            cur4 = self.fc1(x)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4 = self.dropout(spk4)

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            output_spikes.append(spk5)

        output_spikes = torch.stack(output_spikes, dim=0)
        spike_counts = output_spikes.sum(dim=0)

        if return_spikes:
            return spike_counts, output_spikes
        return spike_counts


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_snn_from_ann(snn_model, ann_model):
    ann_state = ann_model.state_dict()
    snn_state = snn_model.state_dict()

    transferable = {}
    for key, value in ann_state.items():
        if key in snn_state and snn_state[key].shape == value.shape:
            transferable[key] = value

    missing, unexpected = snn_model.load_state_dict(transferable, strict=False)
    print(f"Transferred {len(transferable)} tensors from ANN teacher to SNN.")
    print(f"Missing SNN keys after partial load: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")


def distillation_loss(student_counts, teacher_logits, labels, ce_criterion):
    ce = ce_criterion(student_counts, labels)

    t = DISTILL_TEMPERATURE
    student_log_probs = F.log_softmax(student_counts / t, dim=1)
    teacher_probs = F.softmax(teacher_logits / t, dim=1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t * t)

    loss = ce + DISTILL_ALPHA * kd
    return loss, ce.detach(), kd.detach()


def train_one_epoch(snn_model, teacher_model, loader, ce_criterion, optimizer, device):
    snn_model.train()
    teacher_model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(images)

        spike_data = spikegen.rate(images, num_steps=NUM_STEPS)
        spike_counts = snn_model(spike_data)

        loss, ce, kd = distillation_loss(spike_counts, teacher_logits, labels, ce_criterion)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(snn_model.parameters(), max_norm=5.0)
        optimizer.step()

        preds = spike_counts.argmax(dim=1)
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_ce += ce.item() * batch_size
        total_kd += kd.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_ce / total_samples,
        total_kd / total_samples,
        total_correct / total_samples,
    )


@torch.no_grad()
def evaluate(snn_model, loader, ce_criterion, device):
    snn_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        spike_data = spikegen.rate(images, num_steps=NUM_STEPS)
        spike_counts = snn_model(spike_data)
        loss = ce_criterion(spike_counts, labels)
        preds = spike_counts.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Total Loss")
    plt.plot(epochs, history["val_loss"], label="Validation CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SNN-v2 Distillation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "snn_distill_v2_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SNN-v2 Distillation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "snn_distill_v2_accuracy_curve.png", dpi=200)
    plt.close()


@torch.no_grad()
def save_spike_count_sample(snn_model, loader, device):
    snn_model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    spike_data = spikegen.rate(images, num_steps=NUM_STEPS)
    spike_counts, output_spikes = snn_model(spike_data, return_spikes=True)

    sample_idx = 0
    counts = spike_counts[sample_idx].detach().cpu().numpy()
    true_label = labels[sample_idx].item()
    pred_label = int(np.argmax(counts))

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(VOC20_CLASSES)), counts)
    plt.xticks(range(len(VOC20_CLASSES)), VOC20_CLASSES, rotation=60, ha="right")
    plt.xlabel("VOC Class")
    plt.ylabel("Output Spike Count")
    plt.title(
        f"SNN-v2 Output Spike Counts | True: {VOC20_CLASSES[true_label]}, "
        f"Pred: {VOC20_CLASSES[pred_label]}"
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "snn_distill_v2_output_spike_counts_val_sample.png", dpi=200)
    plt.close()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = ROI_DIR / "train.csv"
    val_csv = ROI_DIR / "val.csv"
    teacher_path = MODEL_DIR / "ann_roi_teacher_v2_best.pt"

    if not teacher_path.exists():
        raise FileNotFoundError(f"ANN teacher checkpoint not found: {teacher_path}")

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

    teacher_model = ANNROIClassifier(num_classes=20).to(device)
    teacher_checkpoint = torch.load(teacher_path, map_location=device)
    teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher_model.eval()

    snn_model = SNNROIClassifierV2(num_classes=20, beta=BETA).to(device)
    initialize_snn_from_ann(snn_model, teacher_model)

    print(snn_model)
    print(f"Trainable parameters: {count_parameters(snn_model)}")
    print(f"Rate coding time steps: {NUM_STEPS}")

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(snn_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        "train_loss": [],
        "train_ce": [],
        "train_kd": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = MODEL_DIR / "snn_roi_distill_v2_best.pt"
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_ce, train_kd, train_acc = train_one_epoch(
            snn_model,
            teacher_model,
            train_loader,
            ce_criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(snn_model, val_loader, ce_criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_ce"].append(train_ce)
        history["train_kd"].append(train_kd)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": snn_model.state_dict(),
                "model_name": "SNNROIClassifierV2_Distilled",
                "teacher_checkpoint": str(teacher_path),
                "num_classes": 20,
                "voc20_classes": VOC20_CLASSES,
                "crop_size": CROP_SIZE,
                "beta": BETA,
                "num_steps": NUM_STEPS,
                "distill_alpha": DISTILL_ALPHA,
                "distill_temperature": DISTILL_TEMPERATURE,
                "best_val_acc": best_val_acc,
                "history": history,
            }, best_model_path)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | CE: {train_ce:.4f} | KD: {train_kd:.4f} "
            f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

    total_time = time.time() - start_time

    checkpoint = torch.load(best_model_path, map_location=device)
    snn_model.load_state_dict(checkpoint["model_state_dict"])
    final_val_loss, final_val_acc = evaluate(snn_model, val_loader, ce_criterion, device)

    metrics = {
        "model": "SNN ROI Classifier V2 with ANN-to-SNN initialization and distillation",
        "framework": "CNN proposal generator + proposal-matched ANN teacher + distilled SNN ROI classifier",
        "dataset": "proposal-matched Pascal VOC ROI crops",
        "encoding": "rate coding",
        "neuron_model": "Leaky Integrate-and-Fire",
        "crop_size": CROP_SIZE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "beta": BETA,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "distill_alpha": DISTILL_ALPHA,
        "distill_temperature": DISTILL_TEMPERATURE,
        "best_validation_accuracy": best_val_acc,
        "final_validation_loss": final_val_loss,
        "final_validation_accuracy": final_val_acc,
        "trainable_parameters": count_parameters(snn_model),
        "total_training_time_seconds": total_time,
        "device": str(device),
        "history": history,
    }

    with open(RESULT_DIR / "snn_distill_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    save_curves(history)
    save_spike_count_sample(snn_model, val_loader, device)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {RESULT_DIR / 'snn_distill_v2_metrics.json'}")


if __name__ == "__main__":
    main()
