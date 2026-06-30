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

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF
from torchvision.ops import nms


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
    parser.add_argument("--csv", default="data/processed_300/data_300.csv")
    parser.add_argument("--img-dir", default="data/processed_300/imgs")
    parser.add_argument("--output-dir", default="reports/paper_metrics/experiments")
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[300, 1000, 3000])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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
        cx, cy, width, height, angle = map(float, parts[1:])
        objects.append({"class_id": class_id, "cx": cx, "cy": cy, "w": width, "h": height, "angle": angle})
    return objects


def load_rows(csv_path: Path, img_dir: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["Id"]
            image_path = img_dir / f"{image_id}.jpg"
            if image_path.exists():
                rows.append(
                    {
                        "id": image_id,
                        "clip_id": image_id.rsplit("_", 1)[0],
                        "image_path": image_path,
                        "boxes": parse_target(row["Target"]),
                    }
                )
    rows.sort(key=lambda row: row["id"])
    return rows


def clip_level_indices(rows: list[dict], requested_size: int, val_ratio: float, seed: int) -> tuple[list[int], list[int], int]:
    available = min(requested_size, len(rows))
    selected_indices = list(range(available))
    clip_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx in selected_indices:
        clip_to_indices[rows[idx]["clip_id"]].append(idx)

    clips = list(clip_to_indices)
    random.Random(seed + requested_size).shuffle(clips)
    val_target = max(1, round(available * val_ratio))
    train_indices: list[int] = []
    val_indices: list[int] = []

    for clip_id in clips:
        target = val_indices if len(val_indices) < val_target else train_indices
        target.extend(clip_to_indices[clip_id])

    return train_indices, val_indices, available


class VehicleDataset(Dataset):
    def __init__(self, rows: list[dict], img_size: int, stride: int, train: bool):
        self.rows = rows
        self.img_size = img_size
        self.stride = stride
        self.grid_size = img_size // stride
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def encode_targets(self, boxes: list[dict], orig_w: int, orig_h: int) -> dict[str, torch.Tensor]:
        objectness = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)
        classes = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)
        box_values = torch.zeros((self.grid_size, self.grid_size, 6), dtype=torch.float32)
        area_map = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)
        sx = self.img_size / orig_w
        sy = self.img_size / orig_h

        for box in boxes:
            cx = box["cx"] * sx
            cy = box["cy"] * sy
            width = max(box["w"] * sx, 1.0)
            height = max(box["h"] * sy, 1.0)
            gx = min(max(int(cx / self.stride), 0), self.grid_size - 1)
            gy = min(max(int(cy / self.stride), 0), self.grid_size - 1)
            area = width * height
            if objectness[gy, gx] == 1 and area <= area_map[gy, gx]:
                continue

            objectness[gy, gx] = 1.0
            classes[gy, gx] = box["class_id"] - 1
            area_map[gy, gx] = area
            angle_rad = math.radians(box["angle"])
            box_values[gy, gx] = torch.tensor(
                [
                    cx / self.stride - gx,
                    cy / self.stride - gy,
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
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.train and random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return (
            image_tensor,
            self.encode_targets(row["boxes"], orig_w, orig_h),
            {"id": row["id"], "orig_w": orig_w, "orig_h": orig_h, "gt": row["boxes"]},
        )


def collate_fn(batch):
    images, targets, metas = zip(*batch)
    return (
        torch.stack(images),
        {
            "objectness": torch.stack([target["objectness"] for target in targets]),
            "classes": torch.stack([target["classes"] for target in targets]),
            "boxes": torch.stack([target["boxes"] for target in targets]),
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


class TinyOrientedDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 24, stride=2),
            ConvBlock(24, 48, stride=2),
            ConvBlock(48, 96, stride=2),
            ConvBlock(96, 128, stride=2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 1 + num_classes + 6, kernel_size=1),
        )

    def forward(self, x):
        out = self.head(self.backbone(x))
        out = out.permute(0, 2, 3, 1).contiguous()
        return {"objectness": out[..., 0], "classes": out[..., 1 : 1 + NUM_CLASSES], "boxes": out[..., 1 + NUM_CLASSES :]}


def detection_loss(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], device: torch.device):
    target_obj = target["objectness"].to(device)
    target_cls = target["classes"].to(device)
    target_box = target["boxes"].to(device)
    obj_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))
    cls_loss_fn = nn.CrossEntropyLoss()
    box_loss_fn = nn.SmoothL1Loss()

    obj_loss = obj_loss_fn(pred["objectness"], target_obj)
    pos_mask = target_obj > 0.5
    if pos_mask.any():
        cls_loss = cls_loss_fn(pred["classes"][pos_mask], target_cls[pos_mask])
        box_loss = box_loss_fn(pred["boxes"][pos_mask], target_box[pos_mask])
    else:
        cls_loss = pred["classes"].sum() * 0.0
        box_loss = pred["boxes"].sum() * 0.0

    total = obj_loss + cls_loss + 5.0 * box_loss
    return total, {"total": total.item(), "obj": obj_loss.item(), "cls": cls_loss.item(), "box": box_loss.item()}


def run_epoch(model, loader, device, optimizer=None, max_batches: int = 0):
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
            loss, parts = detection_loss(pred, targets, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = images.size(0)
        for key in totals:
            totals[key] += parts[key] * batch_size
        seen += batch_size
    return {key: value / max(seen, 1) for key, value in totals.items()}


def decode_predictions(pred, metas, img_size, stride, conf_threshold, nms_threshold, max_detections=100):
    obj_prob = torch.sigmoid(pred["objectness"]).cpu()
    class_prob = torch.softmax(pred["classes"], dim=-1).cpu()
    box_pred = pred["boxes"].cpu()
    decoded = []

    for b, meta in enumerate(metas):
        detections = []
        scores, cls_idx = class_prob[b].max(dim=-1)
        scores = scores * obj_prob[b]
        ys, xs = (scores > conf_threshold).nonzero(as_tuple=True)

        for y, x in zip(ys.tolist(), xs.tolist()):
            raw = box_pred[b, y, x]
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
                    "cx": cx_resized * meta["orig_w"] / img_size,
                    "cy": cy_resized * meta["orig_h"] / img_size,
                    "w": width_resized * meta["orig_w"] / img_size,
                    "h": height_resized * meta["orig_h"] / img_size,
                    "angle": angle,
                }
            )

        if detections:
            boxes_xyxy = torch.tensor(
                [[d["cx"] - d["w"] / 2, d["cy"] - d["h"] / 2, d["cx"] + d["w"] / 2, d["cy"] + d["h"] / 2] for d in detections],
                dtype=torch.float32,
            )
            score_tensor = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
            keep = nms(boxes_xyxy, score_tensor, nms_threshold).tolist()[:max_detections]
            detections = [detections[i] for i in keep]
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


def evaluate(model, loader, device, args):
    model.eval()
    per_class_scores = defaultdict(list)
    per_class_matches = defaultdict(list)
    per_class_gt = defaultdict(int)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, _, metas in loader:
            pred = model(images.to(device))
            batch_detections = decode_predictions(pred, metas, args.img_size, args.stride, args.conf_threshold, args.nms_threshold)
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
        "per_class_ap50_obb": {CLASS_NAMES[cls]: per_class_ap[cls] for cls in sorted(CLASS_NAMES)},
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def benchmark_inference(model, loader, device, args):
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
            decode_predictions(pred, metas, args.img_size, args.stride, args.conf_threshold, args.nms_threshold)
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


def experiment(args, rows, train_size, device, output_dir):
    train_indices, val_indices, available_size = clip_level_indices(rows, train_size, args.val_ratio, args.seed)
    train_dataset = Subset(VehicleDataset(rows, args.img_size, args.stride, train=True), train_indices)
    val_dataset = Subset(VehicleDataset(rows, args.img_size, args.stride, train=False), val_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = TinyOrientedDetector(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    history = []
    best_val = float("inf")
    best_state = None
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.max_train_batches)
        val_loss = run_epoch(model, val_loader, device, None, 0)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "seconds": epoch_seconds, "train": train_metrics, "val": val_loss})
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        print(f"size={train_size} epoch={epoch}/{args.epochs} train={train_metrics['total']:.4f} val={val_loss['total']:.4f} seconds={epoch_seconds:.1f}")

    total_train_seconds = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / f"model_subset_{train_size}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    evaluation = evaluate(model, val_loader, device, args)
    inference = benchmark_inference(model, val_loader, device, args)
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    result = {
        "requested_train_size": train_size,
        "available_subset_size": available_size,
        "train_images": len(train_indices),
        "val_images": len(val_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "stride": args.stride,
        "parameters": parameter_count(model),
        "model_size_mb": model_size_mb,
        "total_train_seconds_cpu": total_train_seconds,
        "seconds_per_epoch_cpu": total_train_seconds / max(args.epochs, 1),
        "peak_ram_mb": peak_ram_mb,
        "best_val_loss": best_val,
        **evaluation,
        **inference,
        "checkpoint": str(checkpoint_path),
    }
    write_json(output_dir / f"result_subset_{train_size}.json", result)
    write_json(output_dir / f"history_subset_{train_size}.json", {"history": history})
    return result


def write_paper_summary(output_dir: Path, results: list[dict]) -> None:
    fields = [
        "requested_train_size",
        "available_subset_size",
        "train_images",
        "val_images",
        "epochs",
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
    ]
    write_csv(output_dir / "experiment_summary.csv", results, fields)

    lines = [
        "# Experiment summary for paper",
        "",
        "| Requested subset | Used subset | Train | Val | Macro AP50 OBB | F1 OBB | Latency ms/img CPU | FPS CPU | Train seconds | Model MB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row['requested_train_size']} | {row['available_subset_size']} | {row['train_images']} | {row['val_images']} | "
            f"{row['macro_ap50_obb']:.4f} | {row['f1_obb_iou50']:.4f} | {row['latency_ms_per_image_cpu']:.2f} | "
            f"{row['fps_cpu']:.2f} | {row['total_train_seconds_cpu']:.1f} | {row['model_size_mb']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Use this table to argue the accuracy/compute trade-off for low-resource urban traffic monitoring.",
            "If a requested subset is larger than the local data, `Used subset` documents the cap.",
        ]
    )
    (output_dir / "paper_experiment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.img_size % args.stride != 0:
        raise ValueError("--img-size must be divisible by --stride")

    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.csv), Path(args.img_dir))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []
    for train_size in args.train_sizes:
        print(f"\n=== Experiment subset {train_size} ===")
        result = experiment(args, rows, train_size, device, output_dir)
        results.append(result)
    write_paper_summary(output_dir, results)
    print(f"Wrote experiment artifacts to {output_dir}")


if __name__ == "__main__":
    main()
