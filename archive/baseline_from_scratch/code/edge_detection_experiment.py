#!/usr/bin/env python
"""CPU-oriented vehicle OBB detection experiments for the conference paper.

This script standardizes the PyTorch notebook pipeline into a reproducible
experiment runner. It can train/evaluate the same lightweight detector on
different subset sizes, then exports paper-ready metrics:

- train/validation loss
- oriented-box AP50 proxy per class and macro AP50
- precision/recall/F1 at IoU 0.50
- CPU latency, FPS, model parameters, model size
- per-experiment CSV/JSON summaries

The model is intentionally compact to support the low-resource-device framing.
Raspberry Pi deployment can be left as future work while this script reports
CPU evidence that motivates that direction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import resource
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF


CLASS_NAMES = {
    1: "car",
    2: "van",
    3: "microbus",
    4: "minibus",
    5: "bus",
    6: "articulated_bus",
    7: "truck",
    8: "mototaxi",
    9: "motorcycle",
}
NUM_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Directory containing dataset_1000, ..., dataset_3000.")
    parser.add_argument("--output-dir", default="reports/paper_metrics/experiments")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[1000, 1500, 2000, 2500, 3000])
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs; early stopping can finish sooner.")
    parser.add_argument("--patience", type=int, default=8, help="Stop after this many epochs without validation-F1 improvement.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--strides", nargs="+", type=int, default=[8, 16, 32], help="FPN output strides; image size must be divisible by each.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--pos-weight", type=float, default=8.0, help="Positive weight for objectness; lower values reduce false positives.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Set 0 to use weighted BCE without focal modulation.")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--benchmark-batches", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug option; 0 means use all batches.")
    return parser.parse_args()


def parse_target(target_text: str | None) -> list[dict[str, float]]:
    if target_text is None or target_text.strip().lower() == "none":
        return []
    objects = []
    for raw_object in target_text.split(";"):
        parts = raw_object.strip().split()
        if not parts:
            continue
        if len(parts) != 6:
            raise ValueError(f"Invalid annotation: {raw_object}")
        class_id = int(float(parts[0]))
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unknown class id {class_id} in annotation: {raw_object}")
        cx, cy, width, height, angle = map(float, parts[1:])
        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive box dimensions in annotation: {raw_object}")
        objects.append({"class_id": class_id, "cx": cx, "cy": cy, "w": width, "h": height, "angle": angle % 360.0})
    return objects


def load_rows(csv_path: Path, img_dir: Path) -> list[dict]:
    rows = []
    missing_images = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["Id"]
            image_path = img_dir / f"{image_id}.jpg"
            if not image_path.exists():
                missing_images.append(image_path)
                continue
            rows.append(
                {
                    "id": image_id,
                    "clip_id": image_id.rsplit("_", 1)[0],
                    "image_path": image_path,
                    "boxes": parse_target(row["Target"]),
                }
            )
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} CSV images are missing; first: {missing_images[0]}")
    rows.sort(key=lambda row: row["id"])
    return rows


def clip_level_indices(rows: list[dict], val_ratio: float, test_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Split whole video clips so adjacent frames cannot leak between subsets."""
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and sum to less than one.")
    selected_indices = list(range(len(rows)))
    clip_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx in selected_indices:
        clip_to_indices[rows[idx]["clip_id"]].append(idx)

    clips = list(clip_to_indices)
    random.Random(seed).shuffle(clips)
    val_target = max(1, round(len(rows) * val_ratio))
    test_target = max(1, round(len(rows) * test_ratio))
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for clip_id in clips:
        if len(val_indices) < val_target:
            target = val_indices
        elif len(test_indices) < test_target:
            target = test_indices
        else:
            target = train_indices
        target.extend(clip_to_indices[clip_id])

    return train_indices, val_indices, test_indices


class VehicleDataset(Dataset):
    """Letterboxed images with one OBB target grid per FPN scale."""
    def __init__(self, rows: list[dict], img_size: int, strides: tuple[int, ...], train: bool):
        self.rows = rows
        self.img_size = img_size
        self.strides = strides
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def encode_targets(self, boxes: list[dict], scale: float, pad_x: float, pad_y: float) -> dict[str, torch.Tensor]:
        """Encode boxes after an aspect-ratio-preserving letterbox transform.

        Uniform scaling is deliberate: independently stretching both axes would
        change the geometry of rotated boxes while incorrectly keeping angle.
        """
        objectness = [torch.zeros((self.img_size // stride, self.img_size // stride), dtype=torch.float32) for stride in self.strides]
        classes = [torch.zeros((self.img_size // stride, self.img_size // stride), dtype=torch.long) for stride in self.strides]
        box_values = [torch.zeros((self.img_size // stride, self.img_size // stride, 6), dtype=torch.float32) for stride in self.strides]
        area_map = [torch.zeros((self.img_size // stride, self.img_size // stride), dtype=torch.float32) for stride in self.strides]
        for box in boxes:
            cx = box["cx"] * scale + pad_x
            cy = box["cy"] * scale + pad_y
            width = max(box["w"] * scale, 1.0)
            height = max(box["h"] * scale, 1.0)
            # Small objects use P3/stride 8; medium P4/16; large P5/32.
            level = 0 if max(width, height) < 64 else 1 if max(width, height) < 128 else 2
            stride = self.strides[level]
            grid_size = self.img_size // stride
            gx = min(max(int(cx / stride), 0), grid_size - 1)
            gy = min(max(int(cy / stride), 0), grid_size - 1)
            area = width * height
            if objectness[level][gy, gx] == 1 and area <= area_map[level][gy, gx]:
                continue

            objectness[level][gy, gx] = 1.0
            classes[level][gy, gx] = box["class_id"] - 1
            area_map[level][gy, gx] = area
            angle_rad = math.radians(box["angle"])
            box_values[level][gy, gx] = torch.tensor(
                [
                    cx / stride - gx,
                    cy / stride - gy,
                    math.log(width / self.img_size),
                    math.log(height / self.img_size),
                    math.sin(angle_rad),
                    math.cos(angle_rad),
                ],
                dtype=torch.float32,
            )
        return {"objectness": objectness, "classes": classes, "boxes": box_values}

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        orig_w, orig_h = image.size
        # Letterboxing preserves oriented-box angles and aspect ratios.
        scale = min(self.img_size / orig_w, self.img_size / orig_h)
        resized_w, resized_h = round(orig_w * scale), round(orig_h * scale)
        pad_x, pad_y = (self.img_size - resized_w) // 2, (self.img_size - resized_h) // 2
        image = image.resize((resized_w, resized_h), Image.BILINEAR)
        letterboxed = Image.new("RGB", (self.img_size, self.img_size), (114, 114, 114))
        letterboxed.paste(image, (pad_x, pad_y))
        image = letterboxed
        if self.train and random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return (
            image_tensor,
            self.encode_targets(row["boxes"], scale, pad_x, pad_y),
            {"id": row["id"], "orig_w": orig_w, "orig_h": orig_h, "scale": scale, "pad_x": pad_x, "pad_y": pad_y, "gt": row["boxes"]},
        )


def collate_fn(batch):
    images, targets, metas = zip(*batch)
    return (
        torch.stack(images),
        {
            "objectness": [torch.stack([target["objectness"][level] for target in targets]) for level in range(len(targets[0]["objectness"]))],
            "classes": [torch.stack([target["classes"][level] for target in targets]) for level in range(len(targets[0]["classes"]))],
            "boxes": [torch.stack([target["boxes"][level] for target in targets]) for level in range(len(targets[0]["boxes"]))],
        },
        list(metas),
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DepthwiseBlock(nn.Module):
    """Residual depthwise-separable block: inexpensive enough for tiny CPUs."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm1, self.norm2 = nn.BatchNorm2d(in_channels), nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.act(self.norm2(self.pointwise(self.act(self.norm1(self.depthwise(x))))) + self.skip(x))


class DetectionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(DepthwiseBlock(channels, channels), nn.Conv2d(channels, 1 + num_classes + 6, 1))

    def forward(self, x):
        out = self.net(x).permute(0, 2, 3, 1).contiguous()
        return out


class TinyOrientedDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.stem = ConvBlock(3, 24, stride=2)
        self.s2 = DepthwiseBlock(24, 32, stride=2)
        self.s3 = nn.Sequential(DepthwiseBlock(32, 64, stride=2), DepthwiseBlock(64, 64))
        self.s4 = nn.Sequential(DepthwiseBlock(64, 96, stride=2), DepthwiseBlock(96, 96))
        self.s5 = nn.Sequential(DepthwiseBlock(96, 128, stride=2), DepthwiseBlock(128, 128))
        self.lat3, self.lat4, self.lat5 = nn.Conv2d(64, 64, 1), nn.Conv2d(96, 64, 1), nn.Conv2d(128, 64, 1)
        self.smooth3, self.smooth4 = DepthwiseBlock(64, 64), DepthwiseBlock(64, 64)
        self.heads = nn.ModuleList([DetectionHead(64, num_classes) for _ in range(3)])

    def forward(self, x):
        x = self.s2(self.stem(x)); p3 = self.s3(x); p4 = self.s4(p3); p5 = self.s5(p4)
        f5 = self.lat5(p5)
        f4 = self.smooth4(self.lat4(p4) + F.interpolate(f5, size=p4.shape[-2:], mode="nearest"))
        f3 = self.smooth3(self.lat3(p3) + F.interpolate(f4, size=p3.shape[-2:], mode="nearest"))
        outputs = [head(feature) for head, feature in zip(self.heads, (f3, f4, f5))]
        return {"objectness": [out[..., 0] for out in outputs], "classes": [out[..., 1:1 + self.num_classes] for out in outputs], "boxes": [out[..., 1 + self.num_classes:] for out in outputs]}


def make_loss_config(rows: list[dict], train_indices: list[int], device: torch.device, args):
    """Build loss terms once and counter class imbalance among positive cells."""
    counts = torch.ones(NUM_CLASSES, dtype=torch.float32)
    for idx in train_indices:
        for box in rows[idx]["boxes"]:
            counts[int(box["class_id"]) - 1] += 1
    class_weights = torch.sqrt(counts.sum() / counts)
    class_weights /= class_weights.mean()
    return {
        "pos_weight": torch.tensor(args.pos_weight, device=device),
        "focal_gamma": args.focal_gamma,
        "cls": nn.CrossEntropyLoss(weight=class_weights.to(device)),
        "box": nn.SmoothL1Loss(),
        "class_weights": class_weights.tolist(),
    }


def detection_loss(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], device: torch.device, loss_config: dict):
    target_obj = torch.cat([item.to(device).reshape(-1) for item in target["objectness"]])
    target_cls = torch.cat([item.to(device).reshape(-1) for item in target["classes"]])
    target_box = torch.cat([item.to(device).reshape(-1, 6) for item in target["boxes"]])
    pred_obj = torch.cat([item.reshape(-1) for item in pred["objectness"]])
    pred_cls = torch.cat([item.reshape(-1, NUM_CLASSES) for item in pred["classes"]])
    pred_box = torch.cat([item.reshape(-1, 6) for item in pred["boxes"]])
    raw_obj_loss = F.binary_cross_entropy_with_logits(
        pred_obj, target_obj, pos_weight=loss_config["pos_weight"], reduction="none"
    )
    if loss_config["focal_gamma"] > 0:
        probabilities = torch.sigmoid(pred_obj)
        pt = torch.where(target_obj > 0.5, probabilities, 1 - probabilities)
        raw_obj_loss = (1 - pt).pow(loss_config["focal_gamma"]) * raw_obj_loss
    obj_loss = raw_obj_loss.mean()
    pos_mask = target_obj > 0.5
    if pos_mask.any():
        cls_loss = loss_config["cls"](pred_cls[pos_mask], target_cls[pos_mask])
        box_loss = loss_config["box"](pred_box[pos_mask], target_box[pos_mask])
    else:
        cls_loss = pred_cls.sum() * 0.0
        box_loss = pred_box.sum() * 0.0

    total = obj_loss + cls_loss + 5.0 * box_loss
    return total, {"total": total.item(), "obj": obj_loss.item(), "cls": cls_loss.item(), "box": box_loss.item()}


def run_epoch(model, loader, device, loss_config, optimizer=None, max_batches: int = 0):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"total": 0.0, "obj": 0.0, "cls": 0.0, "box": 0.0}
    seen = 0
    for batch_idx, (images, targets, _) in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break
        images = images.to(device)
        with torch.set_grad_enabled(is_train):
            pred = model(images)
            loss, parts = detection_loss(pred, targets, device, loss_config)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = images.size(0)
        for key in totals:
            totals[key] += parts[key] * batch_size
        seen += batch_size
    return {key: value / max(seen, 1) for key, value in totals.items()}


def oriented_class_aware_nms(detections: list[dict], iou_threshold: float, max_detections: int) -> list[dict]:
    """Suppress only same-class overlapping oriented boxes.

    Axis-aligned, class-agnostic NMS can remove a bus because it overlaps a
    car.  The small candidate set in this CPU baseline makes this exact OBB
    implementation practical and faithful to the evaluation geometry.
    """
    kept: list[dict] = []
    for candidate in sorted(detections, key=lambda item: item["score"], reverse=True):
        if all(candidate["class_id"] != kept_box["class_id"] or oriented_iou(candidate, kept_box) <= iou_threshold for kept_box in kept):
            kept.append(candidate)
            if len(kept) >= max_detections:
                break
    return kept


def decode_predictions(pred, metas, img_size, strides, conf_threshold, nms_threshold, max_detections=100):
    decoded = []

    for b, meta in enumerate(metas):
        detections = []
        for stride, obj_logits, class_logits, boxes in zip(strides, pred["objectness"], pred["classes"], pred["boxes"]):
            scores, cls_idx = torch.softmax(class_logits[b], dim=-1).cpu().max(dim=-1)
            scores = scores * torch.sigmoid(obj_logits[b]).cpu()
            ys, xs = (scores > conf_threshold).nonzero(as_tuple=True)
            for y, x in zip(ys.tolist(), xs.tolist()):
                raw = boxes[b, y, x].cpu()
                x_offset = torch.sigmoid(raw[0]).item()
                y_offset = torch.sigmoid(raw[1]).item()
                width_resized = math.exp(max(min(raw[2].item(), 1.0), -8.0)) * img_size
                height_resized = math.exp(max(min(raw[3].item(), 1.0), -8.0)) * img_size
                cx_resized = (x + x_offset) * stride
                cy_resized = (y + y_offset) * stride
                angle = math.degrees(math.atan2(raw[4].item(), raw[5].item())) % 360.0
                detections.append(
                    {
                        "score": scores[y, x].item(),
                        "class_id": int(cls_idx[y, x].item()) + 1,
                        "cx": (cx_resized - meta["pad_x"]) / meta["scale"],
                        "cy": (cy_resized - meta["pad_y"]) / meta["scale"],
                        "w": width_resized / meta["scale"],
                        "h": height_resized / meta["scale"],
                        "angle": angle,
                    }
                )

        detections = oriented_class_aware_nms(detections, nms_threshold, max_detections)
        decoded.append(detections)
    return decoded


def oriented_box_points(box: dict[str, float]) -> list[tuple[float, float]]:
    cx, cy, width, height = box["cx"], box["cy"], box["w"], box["h"]
    angle = math.radians(box["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = [(-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)]
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2.0


def signed_polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])) / 2.0


def ensure_counter_clockwise(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points if signed_polygon_area(points) >= 0 else list(reversed(points))


def inside(point, edge_start, edge_end) -> bool:
    return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) >= (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])


def intersection(p1, p2, e1, e2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = e1
    x4, y4 = e2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def polygon_clip(subject_polygon, clip_polygon):
    output = subject_polygon
    for i, edge_start in enumerate(clip_polygon):
        edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
        input_list = output
        output = []
        if not input_list:
            break
        previous = input_list[-1]
        for current in input_list:
            if inside(current, edge_start, edge_end):
                if not inside(previous, edge_start, edge_end):
                    output.append(intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif inside(previous, edge_start, edge_end):
                output.append(intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def oriented_iou(a: dict[str, float], b: dict[str, float]) -> float:
    poly_a = ensure_counter_clockwise(oriented_box_points(a))
    poly_b = ensure_counter_clockwise(oriented_box_points(b))
    inter_area = polygon_area(polygon_clip(poly_a, poly_b))
    union = polygon_area(poly_a) + polygon_area(poly_b) - inter_area
    return inter_area / union if union > 0 else 0.0


def average_precision(matches: list[int], scores: list[float], total_gt: int) -> float:
    if total_gt == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for idx in order:
        if matches[idx]:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / total_gt)

    ap = 0.0
    for threshold in [i / 10 for i in range(0, 11)]:
        precision_at_recall = max([p for p, r in zip(precisions, recalls) if r >= threshold] or [0.0])
        ap += precision_at_recall / 11.0
    return ap


def evaluate(model, loader, device, args, conf_threshold: float | None = None, nms_threshold: float | None = None):
    """Evaluate detections at explicitly supplied post-processing thresholds."""
    model.eval()
    conf_threshold = args.conf_threshold if conf_threshold is None else conf_threshold
    nms_threshold = args.nms_threshold if nms_threshold is None else nms_threshold
    per_class_scores = defaultdict(list)
    per_class_matches = defaultdict(list)
    per_class_gt = defaultdict(int)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: list[float] = []

    with torch.no_grad():
        for images, _, metas in loader:
            pred = model(images.to(device))
            batch_detections = decode_predictions(pred, metas, args.img_size, args.strides, conf_threshold, nms_threshold)
            for meta, detections in zip(metas, batch_detections):
                gt_by_class = defaultdict(list)
                matched_by_class = defaultdict(set)
                for gt in meta["gt"]:
                    gt_by_class[gt["class_id"]].append(gt)
                    per_class_gt[gt["class_id"]] += 1

                for det in sorted(detections, key=lambda d: d["score"], reverse=True):
                    cls = det["class_id"]
                    best_iou = 0.0
                    best_idx = -1
                    for idx, gt in enumerate(gt_by_class[cls]):
                        if idx in matched_by_class[cls]:
                            continue
                        iou = oriented_iou(det, gt)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    is_match = best_iou >= 0.5 and best_idx >= 0
                    if is_match:
                        matched_by_class[cls].add(best_idx)
                        total_tp += 1
                        matched_ious.append(best_iou)
                    else:
                        total_fp += 1
                    per_class_scores[cls].append(det["score"])
                    per_class_matches[cls].append(int(is_match))

                for cls, gt_items in gt_by_class.items():
                    total_fn += len(gt_items) - len(matched_by_class[cls])

    per_class_ap = {}
    valid_ap = []
    for cls in sorted(CLASS_NAMES):
        ap = average_precision(per_class_matches[cls], per_class_scores[cls], per_class_gt[cls])
        per_class_ap[cls] = ap
        if not math.isnan(ap):
            valid_ap.append(ap)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision_obb_iou50": precision,
        "recall_obb_iou50": recall,
        "f1_obb_iou50": f1,
        "macro_ap50_obb": sum(valid_ap) / max(len(valid_ap), 1),
        "mean_matched_iou_obb": sum(matched_ious) / max(len(matched_ious), 1),
        "per_class_ap50_obb": {CLASS_NAMES[cls]: per_class_ap[cls] for cls in sorted(CLASS_NAMES)},
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def calibrate_postprocess(model, loader, device, args) -> tuple[float, float, dict]:
    """Choose confidence/NMS only on validation data, never on held-out test data."""
    candidates = []
    for confidence in (0.20, 0.30, 0.40, 0.50, 0.60):
        for nms_iou in (0.30, 0.45, 0.60):
            metrics = evaluate(model, loader, device, args, confidence, nms_iou)
            candidates.append((metrics["f1_obb_iou50"], metrics["macro_ap50_obb"], confidence, nms_iou, metrics))
    # F1 is primary because the observed failure is excessive false positives;
    # AP breaks ties so that ranking quality is still favoured.
    _, _, confidence, nms_iou, metrics = max(candidates, key=lambda item: (item[0], item[1]))
    return confidence, nms_iou, metrics


def benchmark_inference(model, loader, device, args, conf_threshold: float | None = None, nms_threshold: float | None = None):
    model.eval()
    times = []
    images_seen = 0
    with torch.no_grad():
        for batch_idx, (images, _, metas) in enumerate(loader, start=1):
            if batch_idx > args.benchmark_batches:
                break
            images = images.to(device)
            start = time.perf_counter()
            pred = model(images)
            decode_predictions(
                pred, metas, args.img_size, args.strides,
                args.conf_threshold if conf_threshold is None else conf_threshold,
                args.nms_threshold if nms_threshold is None else nms_threshold,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed / images.size(0))
            images_seen += images.size(0)
    mean_latency = sum(times) / max(len(times), 1)
    return {"latency_ms_per_image_cpu": mean_latency * 1000, "fps_cpu": 1 / mean_latency if mean_latency > 0 else 0, "benchmark_images": images_seen}


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_history(output_dir: Path, dataset_size: int, history: list[dict]) -> None:
    """Save a loss curve suitable for the experiment/results section."""
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train"]["total"] for row in history], marker="o", label="Train loss")
    plt.plot(epochs, [row["val"]["total"] for row in history], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Detection loss")
    plt.title(f"Learning curve — dataset_{dataset_size}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"learning_curve_dataset_{dataset_size}.png", dpi=180)
    plt.close()


def plot_per_class_ap(output_dir: Path, dataset_size: int, per_class_ap: dict[str, float]) -> None:
    labels, values = zip(*per_class_ap.items())
    plt.figure(figsize=(9, 4))
    plt.bar(labels, values, color="#3978b5")
    plt.ylim(0, 1)
    plt.ylabel("AP@0.50, oriented IoU")
    plt.title(f"Per-class test AP — dataset_{dataset_size}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"per_class_ap_dataset_{dataset_size}.png", dpi=180)
    plt.close()


def plot_comparison(output_dir: Path, results: list[dict]) -> None:
    """Visual comparison of the two headline paper metrics on held-out test clips."""
    sizes = [str(row["dataset_size"]) for row in results]
    x = list(range(len(sizes)))
    width = 0.36
    plt.figure(figsize=(7, 4))
    plt.bar([item - width / 2 for item in x], [row["macro_ap50_obb"] for row in results], width, label="Macro AP@0.50")
    plt.bar([item + width / 2 for item in x], [row["f1_obb_iou50"] for row in results], width, label="F1@0.50")
    plt.xticks(x, [f"dataset_{size}" for size in sizes])
    plt.ylim(0, 1)
    plt.ylabel("Held-out test score")
    plt.title("Accuracy metrics by dataset size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_metric_comparison.png", dpi=180)
    plt.close()


def experiment(args, rows, dataset_size, device, output_dir):
    """Train on train clips, select by validation loss, report only held-out test metrics."""
    experiment_seed = args.seed + dataset_size
    random.seed(experiment_seed)
    torch.manual_seed(experiment_seed)
    train_indices, val_indices, test_indices = clip_level_indices(rows, args.val_ratio, args.test_ratio, experiment_seed)
    strides = tuple(args.strides)
    train_dataset = Subset(VehicleDataset(rows, args.img_size, strides, train=True), train_indices)
    val_dataset = Subset(VehicleDataset(rows, args.img_size, strides, train=False), val_indices)
    test_dataset = Subset(VehicleDataset(rows, args.img_size, strides, train=False), test_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn, generator=torch.Generator().manual_seed(experiment_seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = TinyOrientedDetector(NUM_CLASSES).to(device)
    loss_config = make_loss_config(rows, train_indices, device, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)

    history = []
    best_val_f1 = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, loss_config, optimizer, args.max_train_batches)
        val_loss = run_epoch(model, val_loader, device, loss_config, None, 0)
        val_detection = evaluate(model, val_loader, device, args)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "seconds": epoch_seconds, "train": train_metrics, "val": val_loss, "val_f1_obb_iou50": val_detection["f1_obb_iou50"], "val_macro_ap50_obb": val_detection["macro_ap50_obb"]})
        scheduler.step(val_detection["f1_obb_iou50"])
        if val_detection["f1_obb_iou50"] > best_val_f1:
            best_val_f1 = val_detection["f1_obb_iou50"]
            best_val_loss = val_loss["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        print(
            f"dataset={dataset_size} epoch={epoch}/{args.epochs} train={train_metrics['total']:.4f} "
            f"val={val_loss['total']:.4f} val_f1={val_detection['f1_obb_iou50']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} seconds={epoch_seconds:.1f}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation F1 was {best_val_f1:.4f} at epoch {best_epoch}.")
            break

    total_train_seconds = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / f"model_dataset_{dataset_size}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    calibrated_confidence, calibrated_nms, validation_evaluation = calibrate_postprocess(model, val_loader, device, args)
    evaluation = evaluate(model, test_loader, device, args, calibrated_confidence, calibrated_nms)
    inference = benchmark_inference(model, test_loader, device, args, calibrated_confidence, calibrated_nms)
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    result = {
        "dataset_size": dataset_size,
        "available_images": len(rows),
        "train_images": len(train_indices),
        "val_images": len(val_indices),
        "test_images": len(test_indices),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "strides": args.strides,
        "parameters": parameter_count(model),
        "model_size_mb": model_size_mb,
        "total_train_seconds_cpu": total_train_seconds,
        "seconds_per_epoch_cpu": total_train_seconds / max(len(history), 1),
        "peak_ram_mb": peak_ram_mb,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1_obb_iou50": best_val_f1,
        "calibrated_conf_threshold": calibrated_confidence,
        "calibrated_nms_threshold": calibrated_nms,
        "pos_weight": args.pos_weight,
        "focal_gamma": args.focal_gamma,
        "class_weights": loss_config["class_weights"],
        **{f"val_{key}": value for key, value in validation_evaluation.items() if key != "per_class_ap50_obb"},
        **evaluation,
        **inference,
        "checkpoint": str(checkpoint_path),
    }
    write_json(output_dir / f"result_dataset_{dataset_size}.json", result)
    write_json(output_dir / f"history_dataset_{dataset_size}.json", {"history": history})
    write_json(output_dir / f"per_class_ap50_dataset_{dataset_size}.json", evaluation["per_class_ap50_obb"])
    plot_history(output_dir, dataset_size, history)
    plot_per_class_ap(output_dir, dataset_size, evaluation["per_class_ap50_obb"])
    return result


def write_paper_summary(output_dir: Path, results: list[dict]) -> None:
    fields = [
        "dataset_size",
        "available_images",
        "train_images",
        "val_images",
        "test_images",
        "epochs_requested",
        "epochs_completed",
        "parameters",
        "model_size_mb",
        "total_train_seconds_cpu",
        "seconds_per_epoch_cpu",
        "latency_ms_per_image_cpu",
        "fps_cpu",
        "peak_ram_mb",
        "precision_obb_iou50",
        "recall_obb_iou50",
        "f1_obb_iou50",
        "macro_ap50_obb",
        "mean_matched_iou_obb",
    ]
    write_csv(output_dir / "experiment_summary.csv", results, fields)

    lines = [
        "# Experiment summary for paper",
        "",
        "| Dataset | Images | Train | Val | Test | Test Macro AP50 OBB | Test F1 OBB | Latency ms/img CPU | FPS CPU | Train seconds | Model MB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row['dataset_size']} | {row['available_images']} | {row['train_images']} | {row['val_images']} | {row['test_images']} | "
            f"{row['macro_ap50_obb']:.4f} | {row['f1_obb_iou50']:.4f} | {row['latency_ms_per_image_cpu']:.2f} | "
            f"{row['fps_cpu']:.2f} | {row['total_train_seconds_cpu']:.1f} | {row['model_size_mb']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Use this table to argue the accuracy/compute trade-off for low-resource urban traffic monitoring.",
            "Scores are calculated once on held-out test clips; validation is used only for checkpoint selection.",
        ]
    )
    (output_dir / "paper_experiment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if any(args.img_size % stride != 0 for stride in args.strides):
        raise ValueError("--img-size must be divisible by every --strides value")
    if args.strides != [8, 16, 32]:
        raise ValueError("This tiny FPN assigns targets to the fixed output strides: 8 16 32.")

    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []
    for dataset_size in args.dataset_sizes:
        dataset_dir = Path(args.data_root) / f"dataset_{dataset_size}"
        csv_path = dataset_dir / f"etiquetas_{dataset_size}.csv"
        img_dir = dataset_dir / "images"
        if not csv_path.exists() or not img_dir.exists():
            raise FileNotFoundError(f"Missing dataset_{dataset_size}: expected {csv_path} and {img_dir}")
        rows = load_rows(csv_path, img_dir)
        if not rows:
            raise ValueError(f"dataset_{dataset_size} contains no valid image/annotation rows")
        print(f"\n=== Experiment dataset_{dataset_size}: {len(rows)} images ===")
        result = experiment(args, rows, dataset_size, device, output_dir)
        results.append(result)
    write_paper_summary(output_dir, results)
    plot_comparison(output_dir, results)
    print(f"Wrote experiment artifacts to {output_dir}")


if __name__ == "__main__":
    main()
