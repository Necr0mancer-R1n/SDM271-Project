import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import snntorch as snn
from snntorch import spikegen

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "cnn_detection"))

from voc_dataset import VOC_CLASSES, IDX_TO_CLASS


TEST_DIR = PROJECT_ROOT / "PascalVOC-Test"

CNN_MODEL_PATH = PROJECT_ROOT / "outputs" / "cnn_detection" / "models" / "cnn_detection_best.pt"
SNN_MODEL_PATH = PROJECT_ROOT / "outputs" / "snn_detection_v1" / "models" / "snn_roi_classifier_best.pt"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "snn_detection_v1"
RESULT_DIR = OUTPUT_ROOT / "results"
FIGURE_DIR = OUTPUT_ROOT / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

VOC20_CLASSES = VOC_CLASSES[1:]


class SNNROIClassifier(nn.Module):
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

        return torch.stack(spk4_rec, dim=0)


def get_cnn_detector(num_classes=21):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def crop_roi(image, box, crop_size=64, padding_ratio=0.05):
    width, height = image.size
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(width, int(x2 + pad_x))
    y2 = min(height, int(y2 + pad_y))

    crop = image.crop((x1, y1, x2, y2))
    crop = crop.resize((crop_size, crop_size))

    return F.to_tensor(crop)


def draw_snn_detections(image, detections, save_path):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = None

    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = box

        class_name = det["snn_class"]
        snn_conf = det["snn_confidence"]
        proposal_score = det["proposal_score"]

        caption = f"{class_name} SNN:{snn_conf:.2f} P:{proposal_score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)

        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        y_text = max(0, y1 - text_h - 4)

        draw.rectangle([x1, y_text, x1 + text_w + 4, y_text + text_h + 4], fill="blue")
        draw.text((x1 + 2, y_text + 2), caption, fill="white", font=font)

    image.save(save_path)


def save_spike_count_heatmap(detections, save_path):
    if len(detections) == 0:
        return

    spike_matrix = np.array([det["spike_counts"] for det in detections], dtype=np.float32)

    plt.figure(figsize=(14, max(4, 0.6 * len(detections))))
    plt.imshow(spike_matrix, aspect="auto")
    plt.colorbar(label="Output Spike Count")

    plt.xticks(range(len(VOC20_CLASSES)), VOC20_CLASSES, rotation=60, ha="right")
    plt.yticks(
        range(len(detections)),
        [f"det{i}: {det['snn_class']}" for i, det in enumerate(detections)]
    )

    plt.xlabel("VOC Class")
    plt.ylabel("Detection ROI")
    plt.title("SNN Output Layer Spike Count Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"PascalVOC-Test folder not found: {TEST_DIR}")

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN detector model not found: {CNN_MODEL_PATH}")

    if not SNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"SNN ROI model not found: {SNN_MODEL_PATH}")

    # Load CNN detector as region proposal generator
    cnn_checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)

    cnn_model = get_cnn_detector(num_classes=len(VOC_CLASSES))
    cnn_model.load_state_dict(cnn_checkpoint["model_state_dict"])
    cnn_model.to(device)
    cnn_model.eval()

    # Load SNN ROI classifier
    snn_checkpoint = torch.load(SNN_MODEL_PATH, map_location=device)

    beta = snn_checkpoint.get("beta", 0.95)
    num_steps = snn_checkpoint.get("num_steps", 20)
    crop_size = snn_checkpoint.get("crop_size", 64)

    snn_model = SNNROIClassifier(num_classes=20, beta=beta)
    snn_model.load_state_dict(snn_checkpoint["model_state_dict"])
    snn_model.to(device)
    snn_model.eval()

    image_paths = sorted([
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found in {TEST_DIR}")

    image_paths = image_paths[:3]

    proposal_threshold = 0.5
    max_detections = 10

    csv_rows = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)

        cnn_output = cnn_model([image_tensor])[0]

        boxes = cnn_output["boxes"].detach().cpu()
        proposal_scores = cnn_output["scores"].detach().cpu()
        cnn_labels = cnn_output["labels"].detach().cpu()

        keep = proposal_scores >= proposal_threshold

        # 如果 0.5 阈值下没有 proposal，则退回取前 5 个低阈值 proposal，避免没有输出。
        if keep.sum().item() == 0:
            top_k = min(5, len(proposal_scores))
            keep_indices = torch.argsort(proposal_scores, descending=True)[:top_k]
        else:
            keep_indices = torch.where(keep)[0]
            keep_indices = keep_indices[:max_detections]

        selected_boxes = boxes[keep_indices]
        selected_scores = proposal_scores[keep_indices]
        selected_cnn_labels = cnn_labels[keep_indices]

        if len(selected_boxes) == 0:
            print(f"{image_path.name}: no candidate boxes")
            continue

        roi_tensors = []

        for box in selected_boxes:
            roi_tensor = crop_roi(
                image,
                box.tolist(),
                crop_size=crop_size,
                padding_ratio=0.05
            )
            roi_tensors.append(roi_tensor)

        roi_batch = torch.stack(roi_tensors, dim=0).to(device)
        spike_data = spikegen.rate(roi_batch, num_steps=num_steps)

        output_spikes = snn_model(spike_data)
        spike_counts = output_spikes.sum(dim=0)

        snn_probs = torch.softmax(spike_counts, dim=1)
        snn_confidences, snn_preds = snn_probs.max(dim=1)

        detections = []

        for i in range(len(selected_boxes)):
            pred_idx = int(snn_preds[i].item())
            snn_class = VOC20_CLASSES[pred_idx]
            snn_conf = float(snn_confidences[i].item())

            cnn_label_idx = int(selected_cnn_labels[i].item())
            cnn_class = IDX_TO_CLASS.get(cnn_label_idx, str(cnn_label_idx))

            box_list = [round(float(x), 2) for x in selected_boxes[i].tolist()]
            counts_list = [float(x) for x in spike_counts[i].detach().cpu().tolist()]

            det = {
                "filename": image_path.name,
                "detection_index": i,
                "box": box_list,
                "proposal_score": float(selected_scores[i].item()),
                "cnn_proposal_class": cnn_class,
                "snn_class": snn_class,
                "snn_confidence": snn_conf,
                "spike_counts": counts_list,
            }

            detections.append(det)

            csv_rows.append({
                "filename": det["filename"],
                "detection_index": det["detection_index"],
                "box": json.dumps(det["box"], ensure_ascii=False),
                "proposal_score": det["proposal_score"],
                "cnn_proposal_class": det["cnn_proposal_class"],
                "snn_class": det["snn_class"],
                "snn_confidence": det["snn_confidence"],
                "spike_counts": json.dumps(det["spike_counts"], ensure_ascii=False),
            })

        detection_figure_path = FIGURE_DIR / f"snn_detection_v1_{image_path.stem}.png"
        spike_figure_path = FIGURE_DIR / f"output_spike_counts_v1_{image_path.stem}.png"

        draw_snn_detections(image.copy(), detections, detection_figure_path)
        save_spike_count_heatmap(detections, spike_figure_path)

        print(f"{image_path.name}: {len(detections)} SNN-v1 detections")

        for det in detections:
            print(
                f"  det{det['detection_index']} "
                f"box={det['box']} "
                f"proposal={det['proposal_score']:.3f} "
                f"cnn_proposal={det['cnn_proposal_class']} "
                f"snn={det['snn_class']} "
                f"snn_conf={det['snn_confidence']:.3f}"
            )

        print(f"  saved detection figure: {detection_figure_path}")
        print(f"  saved spike count heatmap: {spike_figure_path}")

    csv_path = RESULT_DIR / "pascalvoc_test_predictions_v1.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "detection_index",
                "box",
                "proposal_score",
                "cnn_proposal_class",
                "snn_class",
                "snn_confidence",
                "spike_counts",
            ]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved SNN-v1 detection CSV to: {csv_path}")


if __name__ == "__main__":
    main()
