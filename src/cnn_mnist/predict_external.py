import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_ROOT / "MNIST-Test"
MODEL_PATH = PROJECT_ROOT / "outputs" / "cnn_mnist" / "models" / "cnn_mnist_best.pt"
RESULT_DIR = PROJECT_ROOT / "outputs" / "cnn_mnist" / "results"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "cnn_mnist" / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class CNNMnist(nn.Module):
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


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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

    model = CNNMnist().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
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

        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probabilities[0, pred].item()

        results.append({
            "filename": image_path.name,
            "prediction": pred,
            "confidence": confidence
        })

        print(f"{image_path.name} - {pred}")

    csv_path = RESULT_DIR / "mnist_test_predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "prediction", "confidence"]
        )
        writer.writeheader()
        writer.writerows(results)

    # 保存一张外部测试图像预测汇总图
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