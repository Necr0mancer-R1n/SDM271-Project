import json
import time
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import snntorch as snn
from snntorch import spikegen

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "cnn_detection"))

from voc_dataset import VOC_CLASSES, CLASS_TO_IDX, find_voc2007_root


DATA_ROOT = PROJECT_ROOT / "data" / "PascalVOC"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "snn_detection_v1"
MODEL_DIR = OUTPUT_ROOT / "models"
RESULT_DIR = OUTPUT_ROOT / "results"
FIGURE_DIR = OUTPUT_ROOT / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

VOC20_CLASSES = VOC_CLASSES[1:]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VOCCropDataset(Dataset):
    """
    Build an object-crop classification dataset from Pascal VOC annotations.

    Original VOC label:
        background = 0
        aeroplane = 1
        ...
        tvmonitor = 20

    SNN ROI classifier label:
        aeroplane = 0
        ...
        tvmonitor = 19
    """

    def __init__(self, voc_root: Path, image_set: str = "train", crop_size: int = 64, padding_ratio: float = 0.05):
        self.voc_root = Path(voc_root)
        self.image_set = image_set
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio

        split_file = self.voc_root / "ImageSets" / "Main" / f"{image_set}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            image_ids = [line.strip() for line in f.readlines() if line.strip()]

        self.image_dir = self.voc_root / "JPEGImages"
        self.annotation_dir = self.voc_root / "Annotations"

        self.samples = []

        for image_id in image_ids:
            annotation_path = self.annotation_dir / f"{image_id}.xml"
            image_path = self.image_dir / f"{image_id}.jpg"

            if not annotation_path.exists() or not image_path.exists():
                continue

            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for obj_idx, obj in enumerate(root.findall("object")):
                class_name = obj.find("name").text.lower().strip()

                if class_name not in CLASS_TO_IDX:
                    continue

                voc_label = CLASS_TO_IDX[class_name]
                snn_label = voc_label - 1

                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                if xmax <= xmin or ymax <= ymin:
                    continue

                self.samples.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "box": [xmin, ymin, xmax, ymax],
                    "label": snn_label,
                    "class_name": class_name,
                    "obj_idx": obj_idx,
                })

        print(f"{image_set} crop dataset: {len(self.samples)} object crops")

    def __len__(self):
        return len(self.samples)

    def crop_with_padding(self, image, box):
        width, height = image.size
        xmin, ymin, xmax, ymax = box

        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_x = box_w * self.padding_ratio
        pad_y = box_h * self.padding_ratio

        x1 = max(0, int(xmin - pad_x))
        y1 = max(0, int(ymin - pad_y))
        x2 = min(width, int(xmax + pad_x))
        y2 = min(height, int(ymax + pad_y))

        return image.crop((x1, y1, x2, y2))

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample["image_path"]).convert("RGB")
        crop = self.crop_with_padding(image, sample["box"])
        crop = crop.resize((self.crop_size, self.crop_size))

        tensor = F.to_tensor(crop)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return tensor, label


class SNNROIClassifier(nn.Module):
    """
    SNN ROI classifier for Pascal VOC object crops.

    Input:
        Rate-coded ROI sequence, shape [T, B, 3, 64, 64]

    Architecture:
        Conv2d 3 -> 16 + LIF
        MaxPool
        Conv2d 16 -> 32 + LIF
        MaxPool
        FC 32*16*16 -> 256 + LIF
        FC 256 -> 20 + output LIF

    Decision:
        Sum output spikes over time.
        Class with the largest spike count is selected.
    """

    def __init__(self, num_classes=20, beta=0.95):
        super().__init__()

        self.num_classes = num_classes
        self.beta = beta

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.lif3 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta)

    def forward(self, spike_data):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []

        num_steps = spike_data.size(0)

        for step in range(num_steps):
            x = spike_data[step]

            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            x = self.pool1(spk1)

            cur2 = self.conv2(x)
            spk2, mem2 = self.lif2(cur2, mem2)
            x = self.pool2(spk2)

            x = x.flatten(start_dim=1)

            cur3 = self.fc1(x)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk4_rec.append(spk4)

        spk4_rec = torch.stack(spk4_rec, dim=0)
        return spk4_rec


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device, num_steps):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        spike_data = spikegen.rate(images, num_steps=num_steps)

        optimizer.zero_grad()

        output_spikes = model(spike_data)
        spike_counts = output_spikes.sum(dim=0)

        loss = criterion(spike_counts, labels)
        loss.backward()
        optimizer.step()

        preds = spike_counts.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_steps):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        spike_data = spikegen.rate(images, num_steps=num_steps)

        output_spikes = model(spike_data)
        spike_counts = output_spikes.sum(dim=0)

        loss = criterion(spike_counts, labels)
        preds = spike_counts.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_loss_curve(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SNN ROI Classifier Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def save_output_spike_count_sample(model, loader, device, num_steps, save_path):
    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    spike_data = spikegen.rate(images, num_steps=num_steps)
    output_spikes = model(spike_data)
    spike_counts = output_spikes.sum(dim=0)

    sample_index = 0
    counts = spike_counts[sample_index].detach().cpu().numpy()
    true_label = labels[sample_index].item()
    pred_label = int(np.argmax(counts))

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(VOC20_CLASSES)), counts)
    plt.xticks(range(len(VOC20_CLASSES)), VOC20_CLASSES, rotation=60, ha="right")
    plt.xlabel("VOC Class")
    plt.ylabel("Output Spike Count")
    plt.title(
        f"Output Layer Spike Counts | "
        f"True: {VOC20_CLASSES[true_label]}, Pred: {VOC20_CLASSES[pred_label]}"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    voc_root = find_voc2007_root(DATA_ROOT)
    print(f"VOC2007 root: {voc_root}")

    train_dataset = VOCCropDataset(voc_root, image_set="train", crop_size=64)
    val_dataset = VOCCropDataset(voc_root, image_set="val", crop_size=64)

    batch_size = 64
    num_steps = 20
    beta = 0.95
    num_epochs = 8

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

    model = SNNROIClassifier(num_classes=20, beta=beta).to(device)

    print(model)
    print(f"Trainable parameters: {count_parameters(model)}")
    print(f"Rate coding time steps: {num_steps}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    best_model_path = MODEL_DIR / "snn_roi_classifier_best.pt"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_steps
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_steps
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "SNNROIClassifier",
                "num_classes": 20,
                "voc20_classes": VOC20_CLASSES,
                "beta": beta,
                "num_steps": num_steps,
                "crop_size": 64,
                "best_val_acc": best_val_acc,
                "history": history,
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

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device,
        num_steps
    )

    metrics = {
        "model": "SNN ROI Classifier V1",
        "dataset": "Pascal VOC 2007 object crops",
        "framework": "CNN proposal generator + SNN ROI classifier",
        "encoding": "rate coding",
        "neuron_model": "Leaky Integrate-and-Fire",
        "train_split": "VOC2007 train object crops",
        "val_split": "VOC2007 val object crops",
        "num_classes": 20,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "num_steps": num_steps,
        "beta": beta,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "criterion": "CrossEntropyLoss on output spike counts",
        "best_validation_accuracy": best_val_acc,
        "final_validation_loss": val_loss,
        "final_validation_accuracy": val_acc,
        "trainable_parameters": count_parameters(model),
        "total_training_time_seconds": total_time,
        "device": str(device),
        "history": history,
    }

    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    save_loss_curve(history, FIGURE_DIR / "loss_curve.png")

    save_output_spike_count_sample(
        model,
        val_loader,
        device,
        num_steps,
        FIGURE_DIR / "output_spike_counts_val_sample.png"
    )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {RESULT_DIR / 'metrics.json'}")
    print(f"Saved loss curve to: {FIGURE_DIR / 'loss_curve.png'}")
    print(f"Saved output spike count figure to: {FIGURE_DIR / 'output_spike_counts_val_sample.png'}")


if __name__ == "__main__":
    main()
