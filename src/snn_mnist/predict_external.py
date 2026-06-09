import csv
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import snntorch as snn
from snntorch import spikegen

from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_ROOT / "MNIST-Test"
MODEL_PATH = PROJECT_ROOT / "outputs" / "snn_mnist" / "models" / "snn_mnist_best.pt"
RESULT_DIR = PROJECT_ROOT / "outputs" / "snn_mnist" / "results"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "snn_mnist" / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class SNNMnist(nn.Module):
    def __init__(self, input_size=784, hidden_size=1000, output_size=10, beta=0.95):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, spike_data):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        num_steps = spike_data.size(0)

        for step in range(num_steps):
            x = spike_data[step].flatten(start_dim=1)

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        spk2_rec = torch.stack(spk2_rec, dim=0)
        return spk2_rec


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    return image, tensor


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"MNIST-Test folder not found: {TEST_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    hidden_size = checkpoint.get("hidden_size", 1000)
    beta = checkpoint.get("beta", 0.95)
    num_steps = checkpoint.get("num_steps", 25)

    model = SNNMnist(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        beta=beta
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_paths = sorted([
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    ])

    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found in {TEST_DIR}")

    results = []

    for image_path in image_paths:
        original_image, tensor = load_image(image_path)
        tensor = tensor.to(device)

        spike_data = spikegen.rate(tensor, num_steps=num_steps)

        output_spikes = model(spike_data)
        spike_counts = output_spikes.sum(dim=0)

        pred = spike_counts.argmax(dim=1).item()
        confidence = torch.softmax(spike_counts, dim=1)[0, pred].item()

        results.append({
            "filename": image_path.name,
            "prediction": pred,
            "confidence": confidence,
            "output_spike_counts": spike_counts.squeeze(0).detach().cpu().tolist()
        })

        print(f"{image_path.name} - {pred}")

    csv_path = RESULT_DIR / "mnist_test_predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "prediction", "confidence", "output_spike_counts"]
        )
        writer.writeheader()
        writer.writerows(results)

    n = len(image_paths)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(3 * cols, 3 * rows))

    for i, image_path in enumerate(image_paths):
        original_image, _ = load_image(image_path)
        pred = results[i]["prediction"]
        conf = results[i]["confidence"]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(original_image, cmap="gray")
        plt.title(f"{image_path.name}\nPred: {pred}, Conf: {conf:.2f}")
        plt.axis("off")

    plt.tight_layout()
    figure_path = FIGURE_DIR / "mnist_test_predictions.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"Saved predictions to: {csv_path}")
    print(f"Saved prediction figure to: {figure_path}")


if __name__ == "__main__":
    main()