"""
Build a proposal-matched ROI dataset for SNN target detection.

This script uses the trained CNN detector (Faster R-CNN) as a proposal generator.
For each VOC image, predicted proposal boxes are matched to ground-truth boxes by IoU.
Matched proposals are saved as CSV rows, so later ANN/SNN ROI classifiers train on
proposal-like crops rather than only clean ground-truth crops.

Outputs:
    outputs/snn_detection_v2/proposal_rois/train.csv
    outputs/snn_detection_v2/proposal_rois/val.csv
    outputs/snn_detection_v2/proposal_rois/summary.json
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "cnn_detection"))

from voc_dataset import (  # noqa: E402
    VOC_CLASSES,
    IDX_TO_CLASS,
    PascalVOCDataset,
    find_voc2007_root,
    collate_fn,
)


DATA_ROOT = PROJECT_ROOT / "data" / "PascalVOC"
CNN_MODEL_PATH = PROJECT_ROOT / "outputs" / "cnn_detection" / "models" / "cnn_detection_best.pt"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "snn_detection_v2"
ROI_DIR = OUTPUT_ROOT / "proposal_rois"

ROI_DIR.mkdir(parents=True, exist_ok=True)


# Proposal generation settings
SPLITS = ["train", "val"]
BATCH_SIZE = 4
PROPOSAL_SCORE_THRESHOLD = 0.05
MATCH_IOU_THRESHOLD = 0.50
MAX_PROPOSALS_PER_IMAGE = 30
INCLUDE_GT_BOXES = True
SEED = 42


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


def box_iou_torch(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), width - 1.0))
    y1 = max(0.0, min(float(y1), height - 1.0))
    x2 = max(0.0, min(float(x2), width - 1.0))
    y2 = max(0.0, min(float(y2), height - 1.0))
    return [x1, y1, x2, y2]


def add_row(rows, image_path, image_id, box, label_voc, source, proposal_score=1.0, proposal_class="gt", matched_iou=1.0):
    class_name = IDX_TO_CLASS[int(label_voc)]
    label_roi = int(label_voc) - 1  # 0-19 for VOC20 classifier

    rows.append({
        "image_id": image_id,
        "image_path": str(image_path),
        "xmin": round(float(box[0]), 3),
        "ymin": round(float(box[1]), 3),
        "xmax": round(float(box[2]), 3),
        "ymax": round(float(box[3]), 3),
        "label": label_roi,
        "class_name": class_name,
        "source": source,
        "proposal_score": round(float(proposal_score), 6),
        "proposal_class": proposal_class,
        "matched_iou": round(float(matched_iou), 6),
    })


def build_split(split, model, voc_root, device):
    dataset = PascalVOCDataset(voc_root, image_set=split)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    rows = []
    class_counter = Counter()
    source_counter = Counter()

    print(f"Building proposal ROI dataset for split={split}, images={len(dataset)}")

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images_device = [img.to(device) for img in images]
        outputs = model(images_device)

        for image_tensor, output, target in zip(images, outputs, targets):
            image_id = target["image_name"]
            image_path = dataset.image_dir / f"{image_id}.jpg"
            _, height, width = image_tensor.shape

            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu()

            if INCLUDE_GT_BOXES:
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    gt_label_int = int(gt_label.item())
                    if gt_label_int <= 0:
                        continue
                    box = clip_box(gt_box.tolist(), width, height)
                    add_row(
                        rows,
                        image_path,
                        image_id,
                        box,
                        gt_label_int,
                        source="gt",
                        proposal_score=1.0,
                        proposal_class="gt",
                        matched_iou=1.0,
                    )
                    class_counter[IDX_TO_CLASS[gt_label_int]] += 1
                    source_counter["gt"] += 1

            pred_boxes = output["boxes"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            keep = pred_scores >= PROPOSAL_SCORE_THRESHOLD
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            if len(pred_boxes) == 0:
                continue

            order = torch.argsort(pred_scores, descending=True)[:MAX_PROPOSALS_PER_IMAGE]
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]

            ious = box_iou_torch(pred_boxes, gt_boxes)
            best_ious, best_gt_idx = ious.max(dim=1)

            for prop_box, prop_score, prop_label, best_iou, gt_idx in zip(
                pred_boxes, pred_scores, pred_labels, best_ious, best_gt_idx
            ):
                if float(best_iou.item()) < MATCH_IOU_THRESHOLD:
                    continue

                matched_label = int(gt_labels[int(gt_idx.item())].item())
                if matched_label <= 0:
                    continue

                prop_class = IDX_TO_CLASS.get(int(prop_label.item()), str(int(prop_label.item())))
                box = clip_box(prop_box.tolist(), width, height)

                add_row(
                    rows,
                    image_path,
                    image_id,
                    box,
                    matched_label,
                    source="proposal",
                    proposal_score=float(prop_score.item()),
                    proposal_class=prop_class,
                    matched_iou=float(best_iou.item()),
                )

                class_counter[IDX_TO_CLASS[matched_label]] += 1
                source_counter["proposal"] += 1

        if batch_idx % 50 == 0:
            print(f"  processed {batch_idx}/{len(loader)} batches | rows={len(rows)}")

    csv_path = ROI_DIR / f"{split}.csv"
    fieldnames = [
        "image_id",
        "image_path",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "label",
        "class_name",
        "source",
        "proposal_score",
        "proposal_class",
        "matched_iou",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": split,
        "num_images": len(dataset),
        "num_rows": len(rows),
        "source_counts": dict(source_counter),
        "class_counts": dict(class_counter),
        "csv_path": str(csv_path),
    }

    print(f"Saved {split} ROI CSV to: {csv_path}")
    print(f"Source counts: {dict(source_counter)}")
    print(f"Class counts: {dict(class_counter)}")

    return summary


@torch.no_grad()
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    voc_root = find_voc2007_root(DATA_ROOT)
    print(f"VOC2007 root: {voc_root}")

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN detector checkpoint not found: {CNN_MODEL_PATH}")

    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)

    model = get_cnn_detector(num_classes=len(VOC_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    summaries = {}
    for split in SPLITS:
        summaries[split] = build_split(split, model, voc_root, device)

    global_summary = {
        "method": "proposal-matched ROI dataset",
        "cnn_detector": str(CNN_MODEL_PATH),
        "proposal_score_threshold": PROPOSAL_SCORE_THRESHOLD,
        "match_iou_threshold": MATCH_IOU_THRESHOLD,
        "max_proposals_per_image": MAX_PROPOSALS_PER_IMAGE,
        "include_gt_boxes": INCLUDE_GT_BOXES,
        "splits": summaries,
    }

    summary_path = ROI_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=4, ensure_ascii=False)

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
