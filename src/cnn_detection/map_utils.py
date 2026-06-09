import numpy as np


def box_iou_numpy(box_a, box_b):
    if len(box_a) == 0 or len(box_b) == 0:
        return np.zeros((len(box_a), len(box_b)), dtype=np.float32)

    box_a = np.asarray(box_a, dtype=np.float32)
    box_b = np.asarray(box_b, dtype=np.float32)

    lt = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    rb = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, a_min=1e-6, a_max=None)


def voc_2007_ap(recalls, precisions):
    ap = 0.0
    for threshold in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= threshold) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= threshold])
        ap += p / 11.0
    return ap


def compute_voc_map(predictions, ground_truths, num_classes=21, iou_threshold=0.5):
    """
    predictions:
        list of {
            image_id,
            boxes: np.ndarray [N, 4],
            labels: np.ndarray [N],
            scores: np.ndarray [N]
        }

    ground_truths:
        list of {
            image_id,
            boxes: np.ndarray [M, 4],
            labels: np.ndarray [M]
        }
    """

    aps = {}
    class_details = {}

    for class_id in range(1, num_classes):
        class_preds = []
        class_gts = {}

        total_gt = 0

        for gt in ground_truths:
            image_id = gt["image_id"]
            mask = gt["labels"] == class_id
            boxes = gt["boxes"][mask]

            class_gts[image_id] = {
                "boxes": boxes,
                "detected": np.zeros(len(boxes), dtype=bool),
            }
            total_gt += len(boxes)

        for pred in predictions:
            image_id = pred["image_id"]
            mask = pred["labels"] == class_id

            boxes = pred["boxes"][mask]
            scores = pred["scores"][mask]

            for box, score in zip(boxes, scores):
                class_preds.append({
                    "image_id": image_id,
                    "box": box,
                    "score": score,
                })

        if total_gt == 0:
            aps[class_id] = None
            continue

        class_preds = sorted(class_preds, key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            image_id = pred["image_id"]
            pred_box = pred["box"][None, :]

            gt_info = class_gts.get(image_id, {"boxes": np.zeros((0, 4)), "detected": np.zeros(0, dtype=bool)})
            gt_boxes = gt_info["boxes"]

            if len(gt_boxes) == 0:
                fp[i] = 1
                continue

            ious = box_iou_numpy(pred_box, gt_boxes).squeeze(0)
            max_iou_idx = int(np.argmax(ious))
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_threshold and not gt_info["detected"][max_iou_idx]:
                tp[i] = 1
                gt_info["detected"][max_iou_idx] = True
            else:
                fp[i] = 1

        if len(class_preds) == 0:
            aps[class_id] = 0.0
            class_details[class_id] = {
                "ap": 0.0,
                "num_gt": int(total_gt),
                "num_predictions": 0,
            }
            continue

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / max(total_gt, 1)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)

        ap = voc_2007_ap(recalls, precisions)

        aps[class_id] = float(ap)
        class_details[class_id] = {
            "ap": float(ap),
            "num_gt": int(total_gt),
            "num_predictions": int(len(class_preds)),
        }

    valid_aps = [ap for ap in aps.values() if ap is not None]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

    return mean_ap, class_details