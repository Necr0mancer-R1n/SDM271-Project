from pathlib import Path

from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "external" / "MNIST-Test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = datasets.MNIST(
    root=str(DATA_ROOT),
    train=False,
    download=True,
    transform=None
)

used_digits = set()
saved_count = 0

for image, label in dataset:
    if label not in used_digits:
        save_path = OUTPUT_DIR / f"digit_{label}.png"
        image.save(save_path)

        print(f"Saved: {save_path.name} | label: {label}")

        used_digits.add(label)
        saved_count += 1

    if saved_count == 10:
        break

print(f"Done. Saved {saved_count} images to {OUTPUT_DIR}")