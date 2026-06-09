import csv
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from voc_dataset import VOC_CLASSES, IDX_TO_CLASS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_ROOT / "PascalVOC-Test"
MODEL_PATH = PROJECT_ROOT / "outputs" / "cnn_detection" / "models" / "cnn_detection_best.pt"
RESULT_DIR = PROJECT_ROOT / "outputs" / "cnn_detection" / "results"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "cnn_detection" / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def get_model(num_classes=21):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def draw_detections(image, boxes, labels, scores, score_threshold=0.5):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = None

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        class_name = IDX_TO_CLASS.get(int(label), str(int(label)))
        caption = f"{class_name}: {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="red")
        draw.text((x1 + 2, y1 - text_h - 2), caption, fill="white", font=font)

    return image


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"PascalVOC-Test folder not found: {TEST_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = get_model(num_classes=len(VOC_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_paths = sorted([
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found in {TEST_DIR}")

    # 项目要求 3 张图；如果文件夹超过 3 张，默认取前 3 张。
    image_paths = image_paths[:3]

    rows = []

    score_threshold = 0.5

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)

        output = model([image_tensor])[0]

        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scores = output["scores"].detach().cpu()

        keep = scores >= score_threshold

        drawn = draw_detections(
            image.copy(),
            boxes,
            labels,
            scores,
            score_threshold=score_threshold
        )

        save_path = FIGURE_DIR / f"detection_{image_path.stem}.png"
        drawn.save(save_path)

        detection_items = []

        for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
            class_name = IDX_TO_CLASS.get(int(label), str(int(label)))
            box_list = [round(float(x), 2) for x in box.tolist()]

            detection_items.append(
                f"{class_name} {float(score):.3f} {box_list}"
            )

        rows.append({
            "filename": image_path.name,
            "num_detections": int(keep.sum().item()),
            "detections": " | ".join(detection_items),
            "figure_path": str(save_path),
        })

        print(f"{image_path.name}: {int(keep.sum().item())} detections")
        for item in detection_items:
            print(f"  {item}")

    csv_path = RESULT_DIR / "pascalvoc_test_predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "num_detections", "detections", "figure_path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved prediction CSV to: {csv_path}")
    print(f"Saved detection figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()