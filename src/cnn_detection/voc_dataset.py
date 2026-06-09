from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def find_voc2007_root(data_root: Path) -> Path:
    candidates = [
        data_root / "VOCdevkit" / "VOC2007",
        data_root / "VOC2007",
        data_root,
    ]

    for candidate in candidates:
        if (
            (candidate / "Annotations").exists()
            and (candidate / "JPEGImages").exists()
            and (candidate / "ImageSets" / "Main").exists()
        ):
            return candidate

    matches = list(data_root.rglob("VOC2007"))
    for candidate in matches:
        if (
            (candidate / "Annotations").exists()
            and (candidate / "JPEGImages").exists()
            and (candidate / "ImageSets" / "Main").exists()
        ):
            return candidate

    raise FileNotFoundError(
        f"Cannot find VOC2007 structure under {data_root}. "
        "Expected Annotations/, JPEGImages/, ImageSets/Main/."
    )


class PascalVOCDataset(Dataset):
    def __init__(self, voc_root: Path, image_set: str = "trainval"):
        self.voc_root = Path(voc_root)
        self.image_set = image_set

        split_file = self.voc_root / "ImageSets" / "Main" / f"{image_set}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            self.image_ids = [line.strip() for line in f.readlines() if line.strip()]

        self.image_dir = self.voc_root / "JPEGImages"
        self.annotation_dir = self.voc_root / "Annotations"

    def __len__(self):
        return len(self.image_ids)

    def parse_annotation(self, annotation_path: Path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []
        difficult = []

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

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        difficult = torch.as_tensor(difficult, dtype=torch.int64)

        return boxes, labels, difficult

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = self.image_dir / f"{image_id}.jpg"
        annotation_path = self.annotation_dir / f"{image_id}.xml"

        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image)

        boxes, labels, difficult = self.parse_annotation(annotation_path)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": difficult,
            "image_name": image_id,
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))