import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import snntorch as snn
from snntorch import spikegen

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "snn_mnist"
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


class SNNMnist(nn.Module):
    """
    SNN-MNIST classifier.

    Encoding:
        Rate coding. Pixel intensity in [0, 1] is converted to spike probability.

    Network:
        Input: 784 spike neurons
        FC1: 784 -> hidden_size
        LIF1: Leaky integrate-and-fire layer
        FC2: hidden_size -> 10
        LIF2: output LIF layer

    Output:
        Spike counts over time are used as class scores.
    """

    def __init__(self, input_size=784, hidden_size=1000, output_size=10, beta=0.95):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.beta = beta

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, spike_data, return_hidden=False):
        """
        spike_data shape:
            [num_steps, batch_size, 1, 28, 28]
        """

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1_rec = []
        spk2_rec = []

        num_steps = spike_data.size(0)

        for step in range(num_steps):
            x = spike_data[step].flatten(start_dim=1)

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            if return_hidden:
                spk1_rec.append(spk1)

            spk2_rec.append(spk2)

        spk2_rec = torch.stack(spk2_rec, dim=0)

        if return_hidden:
            spk1_rec = torch.stack(spk1_rec, dim=0)
            return spk2_rec, spk1_rec

        return spk2_rec


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

        # Rate coding: [B, 1, 28, 28] -> [T, B, 1, 28, 28]
        spike_data = spikegen.rate(images, num_steps=num_steps)

        optimizer.zero_grad()

        output_spikes = model(spike_data)

        # Spike count over time as classification logits
        spike_counts = output_spikes.sum(dim=0)

        loss = criterion(spike_counts, labels)
        loss.backward()
        optimizer.step()

        preds = spike_counts.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


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

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def save_loss_curve(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SNN-MNIST Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def save_spike_activity_figure(model, loader, device, num_steps, save_path):
    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    spike_data = spikegen.rate(images, num_steps=num_steps)

    output_spikes, hidden_spikes = model(spike_data, return_hidden=True)

    # 使用 batch 中第一个样本
    sample_index = 0
    true_label = labels[sample_index].item()

    output_counts = output_spikes[:, sample_index, :].sum(dim=0)
    pred_label = output_counts.argmax().item()

    # hidden_spikes: [T, B, hidden_size]
    hidden_sample = hidden_spikes[:, sample_index, :128].detach().cpu()

    spike_times = []
    neuron_ids = []

    for t in range(hidden_sample.size(0)):
        active_neurons = torch.where(hidden_sample[t] > 0)[0]
        for neuron_id in active_neurons:
            spike_times.append(t)
            neuron_ids.append(neuron_id.item())

    plt.figure(figsize=(10, 5))
    plt.scatter(spike_times, neuron_ids, s=4)
    plt.xlabel("Time Step")
    plt.ylabel("Hidden Neuron Index")
    plt.title(f"Hidden Layer Spike Raster | Label: {true_label}, Pred: {pred_label}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    # 额外保存输出层 spike count bar 图
    count_path = save_path.parent / "output_spike_counts.png"

    plt.figure(figsize=(7, 4))
    plt.bar(range(10), output_counts.detach().cpu().numpy())
    plt.xlabel("Class")
    plt.ylabel("Spike Count")
    plt.title(f"Output Layer Spike Counts | Label: {true_label}, Pred: {pred_label}")
    plt.xticks(range(10))
    plt.tight_layout()
    plt.savefig(count_path, dpi=200)
    plt.close()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 注意：SNN 速率编码需要像素保持在 [0, 1]，不要做 Normalize。
    transform = transforms.Compose([
        transforms.ToTensor()
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

    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [50000, 10000],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 128
    num_steps = 25
    hidden_size = 1000
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = SNNMnist(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        beta=beta
    ).to(device)

    print(model)
    print(f"Trainable parameters: {count_parameters(model)}")
    print(f"Rate coding time steps: {num_steps}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_path = MODEL_DIR / "snn_mnist_best.pt"

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
                "model_name": "SNNMnist",
                "hidden_size": hidden_size,
                "beta": beta,
                "num_steps": num_steps,
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

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
        num_steps
    )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Total training time: {total_time:.2f}s")

    metrics = {
        "model": "SNNMnist",
        "dataset": "MNIST",
        "encoding": "rate coding",
        "neuron_model": "Leaky Integrate-and-Fire",
        "split": {
            "train": 50000,
            "validation": 10000,
            "test": 10000
        },
        "batch_size": batch_size,
        "epochs": num_epochs,
        "num_steps": num_steps,
        "hidden_size": hidden_size,
        "beta": beta,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "criterion": "CrossEntropyLoss on output spike counts",
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

    save_spike_activity_figure(
        model,
        test_loader,
        device,
        num_steps,
        FIGURE_DIR / "hidden_spike_raster.png"
    )

    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {RESULT_DIR / 'metrics.json'}")
    print(f"Saved loss curve to: {FIGURE_DIR / 'loss_curve.png'}")
    print(f"Saved hidden spike raster to: {FIGURE_DIR / 'hidden_spike_raster.png'}")
    print(f"Saved output spike counts to: {FIGURE_DIR / 'output_spike_counts.png'}")


if __name__ == "__main__":
    main()