import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
import matplotlib.pyplot as plt

import snntorch as snn
from snntorch import spikegen

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
ANN_MODEL_PATH = MODEL_DIR / "ann_roi_teacher_v3_best.pt"

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
        if self.augment and random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
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


class SNNRoiStudentV3(nn.Module):
    def __init__(self, num_classes=21, beta=0.95):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.lif2 = snn.Leaky(beta=beta)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.lif3 = snn.Leaky(beta=beta)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.lif4 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(512, num_classes)
        self.lif5 = snn.Leaky(beta=beta)

    def forward(self, spike_data, return_mem=False):
        mem1 = self.lif1.init_leaky(); mem2 = self.lif2.init_leaky(); mem3 = self.lif3.init_leaky(); mem4 = self.lif4.init_leaky(); mem5 = self.lif5.init_leaky()
        spk5_rec, mem5_rec = [], []
        for step in range(spike_data.size(0)):
            x = spike_data[step]
            cur1 = self.conv1(x); spk1, mem1 = self.lif1(cur1, mem1); x = self.pool1(spk1)
            cur2 = self.conv2(x); spk2, mem2 = self.lif2(cur2, mem2); x = self.pool2(spk2)
            cur3 = self.conv3(x); spk3, mem3 = self.lif3(cur3, mem3)
            x = self.avgpool(spk3).flatten(start_dim=1)
            cur4 = self.fc1(x); spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc2(spk4); spk5, mem5 = self.lif5(cur5, mem5)
            spk5_rec.append(spk5); mem5_rec.append(mem5)
        spk5_rec = torch.stack(spk5_rec, dim=0)
        if return_mem:
            return spk5_rec, torch.stack(mem5_rec, dim=0)
        return spk5_rec


def init_snn_from_ann(snn_model, ann_model):
    snn_model.conv1.weight.data.copy_(ann_model.features[0].weight.data)
    snn_model.conv2.weight.data.copy_(ann_model.features[4].weight.data)
    snn_model.conv3.weight.data.copy_(ann_model.features[8].weight.data)
    snn_model.fc1.weight.data.copy_(ann_model.classifier[1].weight.data)
    snn_model.fc1.bias.data.copy_(ann_model.classifier[1].bias.data)
    snn_model.fc2.weight.data.copy_(ann_model.classifier[4].weight.data)
    snn_model.fc2.bias.data.copy_(ann_model.classifier[4].bias.data)


def compute_class_weights(dataset, num_classes=21, max_weight=3.0):
    labels = [row["label"] for row in dataset.rows]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.25, max_weight)
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int).tolist()


def distillation_loss(student_logits, teacher_logits, temperature=4.0):
    s_log_prob = nnf.log_softmax(student_logits / temperature, dim=1)
    t_prob = nnf.softmax(teacher_logits / temperature, dim=1)
    return nnf.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temperature ** 2)


def train_one_epoch(snn_model, ann_teacher, loader, ce_criterion, optimizer, device, num_steps, alpha, temperature, mem_loss_weight):
    snn_model.train(); ann_teacher.eval()
    total_loss = total_ce = total_kd = total_mem = total_correct = total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            teacher_logits = ann_teacher(images)
        spike_data = spikegen.rate(images, num_steps=num_steps)
        output_spikes, output_mems = snn_model(spike_data, return_mem=True)
        spike_counts = output_spikes.sum(dim=0)
        mean_mem = output_mems.mean(dim=0)
        ce_loss = ce_criterion(spike_counts, labels)
        kd_loss = distillation_loss(spike_counts, teacher_logits, temperature=temperature)
        mem_loss = ce_criterion(mean_mem, labels)
        loss = (1.0 - alpha) * ce_loss + alpha * kd_loss + mem_loss_weight * mem_loss
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(snn_model.parameters(), max_norm=5.0); optimizer.step()
        preds = spike_counts.argmax(dim=1)
        total_loss += loss.item() * images.size(0); total_ce += ce_loss.item() * images.size(0); total_kd += kd_loss.item() * images.size(0); total_mem += mem_loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item(); total_samples += images.size(0)
    return {"loss": total_loss/total_samples, "ce": total_ce/total_samples, "kd": total_kd/total_samples, "mem": total_mem/total_samples, "accuracy": total_correct/total_samples}


@torch.no_grad()
def evaluate(snn_model, loader, ce_criterion, device, num_steps, num_repeats=1):
    snn_model.eval()
    total_loss = total_correct = total_samples = 0
    non_bg_correct = non_bg_total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        spike_counts_sum = None
        for _ in range(num_repeats):
            spike_data = spikegen.rate(images, num_steps=num_steps)
            output_spikes = snn_model(spike_data)
            spike_counts = output_spikes.sum(dim=0)
            spike_counts_sum = spike_counts if spike_counts_sum is None else spike_counts_sum + spike_counts
        spike_counts_avg = spike_counts_sum / float(num_repeats)
        loss = ce_criterion(spike_counts_avg, labels)
        preds = spike_counts_avg.argmax(dim=1)
        total_loss += loss.item() * images.size(0); total_correct += (preds == labels).sum().item(); total_samples += images.size(0)
        non_bg = labels != 0
        non_bg_correct += ((preds == labels) & non_bg).sum().item(); non_bg_total += non_bg.sum().item()
    return {"loss": total_loss/total_samples, "accuracy": total_correct/total_samples, "non_background_accuracy": non_bg_correct/max(non_bg_total,1)}


def freeze_feature_layers(model, freeze=True):
    for layer in [model.conv1, model.conv2, model.conv3]:
        for p in layer.parameters():
            p.requires_grad = not freeze


def save_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8,5)); plt.plot(epochs, history["train_loss"], label="Train Loss"); plt.plot(epochs, history["val_loss"], label="Val Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("SNN ROI Distillation v3 Loss"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(FIGURE_DIR / "snn_distill_v3_loss_curve.png", dpi=200); plt.close()
    plt.figure(figsize=(8,5)); plt.plot(epochs, history["train_acc"], label="Train Acc"); plt.plot(epochs, history["val_acc"], label="Val Acc"); plt.plot(epochs, history["val_non_bg_acc"], label="Val Non-bg Acc"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("SNN ROI Distillation v3 Accuracy"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(FIGURE_DIR / "snn_distill_v3_accuracy_curve.png", dpi=200); plt.close()


@torch.no_grad()
def save_spike_count_sample(snn_model, loader, device, num_steps, save_path):
    snn_model.eval(); images, labels = next(iter(loader)); images, labels = images.to(device), labels.to(device)
    spike_data = spikegen.rate(images, num_steps=num_steps); output_spikes = snn_model(spike_data); spike_counts = output_spikes.sum(dim=0)
    sample_idx = 0; counts = spike_counts[sample_idx].detach().cpu().numpy(); true_label = int(labels[sample_idx].item()); pred_label = int(np.argmax(counts))
    plt.figure(figsize=(12,5)); plt.bar(range(NUM_CLASSES), counts); plt.xticks(range(NUM_CLASSES), VOC_CLASSES, rotation=60, ha="right"); plt.xlabel("Class"); plt.ylabel("Output Spike Count"); plt.title(f"SNN-v3 Output Spike Counts | True: {VOC_CLASSES[true_label]}, Pred: {VOC_CLASSES[pred_label]}"); plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not ANN_MODEL_PATH.exists():
        raise FileNotFoundError(f"ANN teacher checkpoint not found: {ANN_MODEL_PATH}")
    train_dataset = RoiCsvDataset(ROI_DIR / "train.csv", crop_size=CROP_SIZE, padding_ratio=PADDING_RATIO, augment=True)
    val_dataset = RoiCsvDataset(ROI_DIR / "val.csv", crop_size=CROP_SIZE, padding_ratio=PADDING_RATIO, augment=False)
    class_weights, class_counts = compute_class_weights(train_dataset, num_classes=NUM_CLASSES, max_weight=3.0)
    print("Class counts:", class_counts); print("Class weights:", [round(float(x),4) for x in class_weights.tolist()])
    batch_size, num_steps, beta, num_epochs = 64, 30, 0.95, 15
    alpha, temperature, mem_loss_weight, eval_repeats = 0.75, 4.0, 0.10, 3
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ann_teacher = AnnRoiTeacherV3(num_classes=NUM_CLASSES).to(device)
    ann_ckpt = torch.load(ANN_MODEL_PATH, map_location=device)
    ann_teacher.load_state_dict(ann_ckpt["model_state_dict"]); ann_teacher.eval()
    snn_model = SNNRoiStudentV3(num_classes=NUM_CLASSES, beta=beta).to(device)
    init_snn_from_ann(snn_model, ann_teacher)
    print(snn_model); print("Initialized SNN Conv/FC weights from ANN teacher.")
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.01)
    optimizer = optim.AdamW(snn_model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = 0.0; best_model_path = MODEL_DIR / "snn_roi_distill_v3_best.pt"
    history = {"train_loss": [], "train_ce": [], "train_kd": [], "train_mem": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_non_bg_acc": []}
    import time
    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        epoch_start = time.time()
        if epoch <= 3:
            freeze_feature_layers(snn_model, freeze=True)
            if epoch == 1: print("Epoch 1-3: freezing SNN conv layers for warm-up.")
        else:
            freeze_feature_layers(snn_model, freeze=False)
            if epoch == 4: print("Epoch 4+: unfreezing all SNN layers.")
        train_metrics = train_one_epoch(snn_model, ann_teacher, train_loader, ce_criterion, optimizer, device, num_steps, alpha, temperature, mem_loss_weight)
        val_metrics = evaluate(snn_model, val_loader, ce_criterion, device, num_steps, num_repeats=eval_repeats)
        scheduler.step()
        history["train_loss"].append(train_metrics["loss"]); history["train_ce"].append(train_metrics["ce"]); history["train_kd"].append(train_metrics["kd"]); history["train_mem"].append(train_metrics["mem"]); history["train_acc"].append(train_metrics["accuracy"]); history["val_loss"].append(val_metrics["loss"]); history["val_acc"].append(val_metrics["accuracy"]); history["val_non_bg_acc"].append(val_metrics["non_background_accuracy"])
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"model_state_dict": snn_model.state_dict(), "model_name":"SNNRoiStudentV3", "num_classes": NUM_CLASSES, "class_names": VOC_CLASSES, "beta": beta, "num_steps": num_steps, "crop_size": CROP_SIZE, "padding_ratio": PADDING_RATIO, "alpha": alpha, "temperature": temperature, "mem_loss_weight": mem_loss_weight, "eval_repeats": eval_repeats, "best_validation_accuracy": best_val_acc, "history": history}, best_model_path)
        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {train_metrics['loss']:.4f} (CE {train_metrics['ce']:.4f}, KD {train_metrics['kd']:.4f}, MEM {train_metrics['mem']:.4f}) | Train Acc: {train_metrics['accuracy']:.4f} Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} Val Non-bg Acc: {val_metrics['non_background_accuracy']:.4f} Time: {time.time()-epoch_start:.2f}s")
    total_time = time.time() - start_time
    checkpoint = torch.load(best_model_path, map_location=device); snn_model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate(snn_model, val_loader, ce_criterion, device, num_steps, num_repeats=eval_repeats)
    metrics = {"model":"SNNRoiStudentV3", "dataset":"SNN detection v3 ROI dataset", "num_classes":NUM_CLASSES, "class_names":VOC_CLASSES, "crop_size":CROP_SIZE, "padding_ratio":PADDING_RATIO, "epochs":num_epochs, "batch_size":batch_size, "num_steps":num_steps, "beta":beta, "alpha":alpha, "temperature":temperature, "mem_loss_weight":mem_loss_weight, "eval_repeats":eval_repeats, "optimizer":"AdamW", "learning_rate":3e-4, "weight_decay":1e-4, "class_counts":class_counts, "class_weights":[float(x) for x in class_weights.tolist()], "best_validation_accuracy":best_val_acc, "final_validation_loss":final_metrics["loss"], "final_validation_accuracy":final_metrics["accuracy"], "final_non_background_accuracy":final_metrics["non_background_accuracy"], "total_training_time_seconds":total_time, "history":history}
    with open(RESULT_DIR / "snn_distill_v3_metrics.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=4, ensure_ascii=False)
    save_curves(history); save_spike_count_sample(snn_model, val_loader, device, num_steps, FIGURE_DIR / "snn_v3_output_spike_counts_val_sample.png")
    print(f"Best validation accuracy: {best_val_acc:.4f}"); print(f"Final validation loss: {final_metrics['loss']:.4f}"); print(f"Final validation accuracy: {final_metrics['accuracy']:.4f}"); print(f"Final non-background accuracy: {final_metrics['non_background_accuracy']:.4f}"); print(f"Total training time: {total_time:.2f}s"); print(f"Saved best model to: {best_model_path}"); print(f"Saved metrics to: {RESULT_DIR / 'snn_distill_v3_metrics.json'}")


if __name__ == "__main__":
    main()
