import csv
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "cnn_detection"))

from voc_dataset import VOC_CLASSES, CLASS_TO_IDX, find_voc2007_root


DATA_ROOT = PROJECT_ROOT / "data" / "PascalVOC"
CNN_MODEL_PATH = PROJECT_ROOT / "outputs" / "cnn_detection" / "models" / "cnn_detection_best.pt"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "snn_detection_v3"
ROI_DIR = OUTPUT_ROOT / "roi_dataset"
ROI_DIR.mkdir(parents=True, exist_ok=True)

POS_IOU_THRESHOLD = 0.60
NEG_IOU_THRESHOLD = 0.30
PROPOSAL_SCORE_THRESHOLD = 0.05
MAX_PROPOSALS_PER_IMAGE = 80
MAX_BACKGROUND_PER_IMAGE = 8
RANDOM_SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cnn_detector(num_classes=21):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def box_iou_np(boxes1, boxes2):
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0, None)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes, labels, difficult = [], [], []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower().strip()
        if class_name not in CLASS_TO_IDX:
            continue
        diff_tag = obj.find("difficult")
        diff = int(diff_tag.text) if diff_tag is not None else 0
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[class_name])
        difficult.append(diff)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64), np.asarray(difficult, dtype=np.int64)


@torch.no_grad()
def get_proposals(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).to(device)
    output = model([tensor])[0]
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    keep = scores >= PROPOSAL_SCORE_THRESHOLD
    boxes = boxes[keep]
    scores = scores[keep]
    order = np.argsort(-scores)[:MAX_PROPOSALS_PER_IMAGE]
    return boxes[order], scores[order]


def build_split(voc_root, image_set, model, device):
    split_file = voc_root / "ImageSets" / "Main" / f"{image_set}.txt"
    with open(split_file, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]
    rows = []
    stats = {
        "image_set": image_set,
        "num_images": len(image_ids),
        "num_gt_rows": 0,
        "num_positive_proposal_rows": 0,
        "num_background_rows": 0,
        "class_counts": {str(i): 0 for i in range(len(VOC_CLASSES))},
    }
    image_dir = voc_root / "JPEGImages"
    ann_dir = voc_root / "Annotations"
    for idx, image_id in enumerate(image_ids, start=1):
        image_path = image_dir / f"{image_id}.jpg"
        ann_path = ann_dir / f"{image_id}.xml"
        if not image_path.exists() or not ann_path.exists():
            continue
        gt_boxes, gt_labels, difficult = parse_annotation(ann_path)
        if len(gt_boxes) == 0:
            continue
        for box, label in zip(gt_boxes, gt_labels):
            if label <= 0:
                continue
            rows.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "source": "gt",
                "label": int(label),
                "class_name": VOC_CLASSES[int(label)],
                "box": json.dumps([float(x) for x in box]),
                "matched_iou": 1.0,
                "proposal_score": 1.0,
            })
            stats["num_gt_rows"] += 1
            stats["class_counts"][str(int(label))] += 1
        proposal_boxes, proposal_scores = get_proposals(model, image_path, device)
        if len(proposal_boxes) == 0:
            continue
        ious = box_iou_np(proposal_boxes, gt_boxes)
        max_iou = ious.max(axis=1)
        matched_gt_idx = ious.argmax(axis=1)
        pos_indices = np.where(max_iou >= POS_IOU_THRESHOLD)[0]
        neg_indices = np.where(max_iou < NEG_IOU_THRESHOLD)[0]
        for prop_idx in pos_indices:
            gt_idx = matched_gt_idx[prop_idx]
            label = int(gt_labels[gt_idx])
            if label <= 0:
                continue
            box = proposal_boxes[prop_idx]
            rows.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "source": "proposal_pos",
                "label": label,
                "class_name": VOC_CLASSES[label],
                "box": json.dumps([float(x) for x in box]),
                "matched_iou": float(max_iou[prop_idx]),
                "proposal_score": float(proposal_scores[prop_idx]),
            })
            stats["num_positive_proposal_rows"] += 1
            stats["class_counts"][str(label)] += 1
        if len(neg_indices) > 0:
            neg_indices = sorted(neg_indices, key=lambda j: proposal_scores[j], reverse=True)[:MAX_BACKGROUND_PER_IMAGE]
            for prop_idx in neg_indices:
                box = proposal_boxes[prop_idx]
                rows.append({
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "source": "background",
                    "label": 0,
                    "class_name": "background",
                    "box": json.dumps([float(x) for x in box]),
                    "matched_iou": float(max_iou[prop_idx]),
                    "proposal_score": float(proposal_scores[prop_idx]),
                })
                stats["num_background_rows"] += 1
                stats["class_counts"]["0"] += 1
        if idx % 100 == 0:
            print(f"[{image_set}] {idx}/{len(image_ids)} images | rows={len(rows)} gt={stats['num_gt_rows']} pos={stats['num_positive_proposal_rows']} bg={stats['num_background_rows']}")
    return rows, stats


def save_csv(rows, path):
    fieldnames = ["image_id", "image_path", "source", "label", "class_name", "box", "matched_iou", "proposal_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    voc_root = find_voc2007_root(DATA_ROOT)
    print(f"VOC2007 root: {voc_root}")
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN detector model not found: {CNN_MODEL_PATH}")
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
    model = get_cnn_detector(num_classes=len(VOC_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Loaded Faster R-CNN proposal generator.")
    print(f"Positive proposal IoU threshold: {POS_IOU_THRESHOLD}")
    print(f"Background proposal IoU threshold: {NEG_IOU_THRESHOLD}")
    train_rows, train_stats = build_split(voc_root, "train", model, device)
    val_rows, val_stats = build_split(voc_root, "val", model, device)
    save_csv(train_rows, ROI_DIR / "train.csv")
    save_csv(val_rows, ROI_DIR / "val.csv")
    summary = {
        "version": "snn_detection_v3",
        "description": "21-class ROI dataset: background + VOC20; GT crops + high-IoU proposals + background proposals.",
        "positive_iou_threshold": POS_IOU_THRESHOLD,
        "negative_iou_threshold": NEG_IOU_THRESHOLD,
        "proposal_score_threshold": PROPOSAL_SCORE_THRESHOLD,
        "max_proposals_per_image": MAX_PROPOSALS_PER_IMAGE,
        "max_background_per_image": MAX_BACKGROUND_PER_IMAGE,
        "num_classes": len(VOC_CLASSES),
        "class_names": VOC_CLASSES,
        "train": train_stats,
        "val": val_stats,
    }
    with open(ROI_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"Saved train CSV to: {ROI_DIR / 'train.csv'}")
    print(f"Saved val CSV to: {ROI_DIR / 'val.csv'}")
    print(f"Saved summary to: {ROI_DIR / 'summary.json'}")
    print("Train stats:", json.dumps(train_stats, indent=2, ensure_ascii=False))
    print("Val stats:", json.dumps(val_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
