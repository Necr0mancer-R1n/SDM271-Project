import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from voc_dataset import PascalVOCDataset, find_voc2007_root, collate_fn, VOC_CLASSES
from map_utils import compute_voc_map


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "PascalVOC"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cnn_detection"
MODEL_DIR = OUTPUT_ROOT / "models"
RESULT_DIR = OUTPUT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(num_classes=21):
    try:
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        print("Loaded COCO-pretrained Faster R-CNN weights.")
    except Exception as e:
        print(f"Could not load pretrained weights automatically: {e}")
        print("Using randomly initialized Faster R-CNN.")
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    loss_details = {}

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]

        new_targets = []
        for target in targets:
            new_target = {}
            for key, value in target.items():
                if torch.is_tensor(value):
                    new_target[key] = value.to(device)
                elif key != "image_name":
                    new_target[key] = value
            new_targets.append(new_target)

        loss_dict = model(images, new_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        for key, value in loss_dict.items():
            loss_details[key] = loss_details.get(key, 0.0) + value.item()

        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss: {losses.item():.4f}"
            )

    avg_loss = total_loss / len(loader)
    avg_details = {k: v / len(loader) for k, v in loss_details.items()}

    return avg_loss, avg_details


@torch.no_grad()
def evaluate_map(model, loader, device, score_threshold=0.05):
    model.eval()

    predictions = []
    ground_truths = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            image_id = target["image_name"]

            boxes = output["boxes"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()

            keep = scores >= score_threshold

            predictions.append({
                "image_id": image_id,
                "boxes": boxes[keep],
                "labels": labels[keep],
                "scores": scores[keep],
            })

            ground_truths.append({
                "image_id": image_id,
                "boxes": target["boxes"].detach().cpu().numpy(),
                "labels": target["labels"].detach().cpu().numpy(),
            })

    mean_ap, class_details = compute_voc_map(
        predictions,
        ground_truths,
        num_classes=len(VOC_CLASSES),
        iou_threshold=0.5
    )

    return mean_ap, class_details


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    voc_root = find_voc2007_root(DATA_ROOT)
    print(f"VOC2007 root: {voc_root}")

    train_dataset = PascalVOCDataset(voc_root, image_set="trainval")

    # VOC2007 如果有 test.txt 就用 test 评估；没有就用 val。
    split_dir = voc_root / "ImageSets" / "Main"
    eval_split = "test" if (split_dir / "test.txt").exists() else "val"
    eval_dataset = PascalVOCDataset(voc_root, image_set=eval_split)

    print(f"Train split: trainval | images: {len(train_dataset)}")
    print(f"Eval split: {eval_split} | images: {len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = get_model(num_classes=len(VOC_CLASSES))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = 5

    best_map = 0.0
    history = {
        "train_loss": [],
        "map": []
    }

    best_model_path = MODEL_DIR / "cnn_detection_best.pt"

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, loss_details = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch
        )

        lr_scheduler.step()

        mean_ap, class_details = evaluate_map(
            model,
            eval_loader,
            device,
            score_threshold=0.05
        )

        history["train_loss"].append(train_loss)
        history["map"].append(mean_ap)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"mAP@0.5: {mean_ap:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if mean_ap > best_map:
            best_map = mean_ap

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "FasterRCNN_MobileNetV3_FPN",
                "num_classes": len(VOC_CLASSES),
                "voc_classes": VOC_CLASSES,
                "best_map": best_map,
                "epoch": epoch,
                "eval_split": eval_split,
                "history": history,
            }, best_model_path)

    total_time = time.time() - start_time

    results = {
        "model": "Faster R-CNN MobileNetV3-Large FPN",
        "dataset": "Pascal VOC 2007",
        "train_split": "trainval",
        "eval_split": eval_split,
        "num_classes": len(VOC_CLASSES),
        "epochs": num_epochs,
        "batch_size": 4,
        "optimizer": "SGD",
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "metric": "mAP@0.5, Pascal VOC 2007 11-point AP",
        "best_map": best_map,
        "total_training_time_seconds": total_time,
        "history": history,
    }

    with open(RESULT_DIR / "map_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Best mAP@0.5: {best_map:.4f}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved results to: {RESULT_DIR / 'map_results.json'}")


if __name__ == "__main__":
    main()