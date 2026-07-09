#!/usr/bin/env python
"""Dataset audit pipeline for the vehicle OBB conference paper.

This script standardizes the dataset-management work from
``trash/data-management.ipynb`` and produces paper-ready CSV/JSON artifacts:

- dataset integrity summary
- class distribution
- object geometry statistics
- image dimension statistics
- clip-level train/validation split manifests for 300/1000/3000 samples

It intentionally uses only the annotation CSV and image directory passed by
argument, so the same pipeline can be rerun when larger MTC subsets are added.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/processed_300/data_300.csv", help="Annotation CSV with Id,Target columns.")
    parser.add_argument("--img-dir", default="data/processed_300/imgs", help="Directory containing JPG images.")
    parser.add_argument("--output-dir", default="reports/paper_metrics/dataset", help="Directory for generated artifacts.")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[300, 1000, 3000], help="Requested subset sizes.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for split manifests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    return parser.parse_args()


def parse_target(target_text: str | None) -> list[dict[str, float]]:
    if target_text is None or target_text.strip().lower() == "none":
        return []

    objects: list[dict[str, float]] = []
    for raw_object in target_text.split(";"):
        parts = raw_object.strip().split()
        if not parts:
            continue
        if len(parts) != 6:
            raise ValueError(f"Invalid annotation, expected 6 values: {raw_object}")
        class_id = int(float(parts[0]))
        cx, cy, width, height, angle = map(float, parts[1:])
        objects.append(
            {
                "class_id": class_id,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "angle": angle,
            }
        )
    return objects


def load_rows(csv_path: Path, img_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["Id"]
            rows.append(
                {
                    "id": image_id,
                    "clip_id": image_id.rsplit("_", 1)[0],
                    "target": row["Target"],
                    "objects": parse_target(row["Target"]),
                    "image_path": img_dir / f"{image_id}.jpg",
                }
            )
    return rows


def summarize_numbers(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def inspect_images(rows: list[dict]) -> tuple[list[dict], Counter]:
    image_records: list[dict] = []
    dimensions: Counter = Counter()
    for row in rows:
        if not row["image_path"].exists():
            continue
        with Image.open(row["image_path"]) as image:
            width, height = image.size
        row["image_width"] = width
        row["image_height"] = height
        dimensions[(width, height)] += 1
        image_records.append({"id": row["id"], "width": width, "height": height})
    return image_records, dimensions


def object_records(rows: list[dict]) -> list[dict]:
    records: list[dict] = []
    for row in rows:
        image_width = row.get("image_width")
        image_height = row.get("image_height")
        for obj in row["objects"]:
            area = obj["width"] * obj["height"]
            image_area = image_width * image_height if image_width and image_height else None
            out_of_bounds = False
            if image_width and image_height:
                out_of_bounds = (
                    obj["cx"] < 0
                    or obj["cy"] < 0
                    or obj["cx"] > image_width
                    or obj["cy"] > image_height
                    or obj["width"] <= 0
                    or obj["height"] <= 0
                )
            records.append(
                {
                    "image_id": row["id"],
                    "clip_id": row["clip_id"],
                    "class_id": obj["class_id"],
                    "class_name": CLASS_NAMES.get(obj["class_id"], "unknown"),
                    "cx": obj["cx"],
                    "cy": obj["cy"],
                    "width": obj["width"],
                    "height": obj["height"],
                    "angle": obj["angle"],
                    "area_px": area,
                    "area_ratio": area / image_area if image_area else "",
                    "out_of_bounds": int(out_of_bounds),
                }
            )
    return records


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_clip_level_split(rows: list[dict], requested_size: int, val_ratio: float, seed: int) -> list[dict]:
    available = min(requested_size, len(rows))
    selected_rows = rows[:available]

    clip_to_rows: dict[str, list[dict]] = defaultdict(list)
    for row in selected_rows:
        clip_to_rows[row["clip_id"]].append(row)

    clips = list(clip_to_rows)
    random.Random(seed + requested_size).shuffle(clips)
    val_target = max(1, round(available * val_ratio))

    split_by_id: dict[str, str] = {}
    val_count = 0
    for clip_id in clips:
        split = "val" if val_count < val_target else "train"
        for row in clip_to_rows[clip_id]:
            split_by_id[row["id"]] = split
        if split == "val":
            val_count += len(clip_to_rows[clip_id])

    return [
        {
            "id": row["id"],
            "clip_id": row["clip_id"],
            "split": split_by_id[row["id"]],
            "object_count": len(row["objects"]),
            "available_subset_size": available,
            "requested_subset_size": requested_size,
        }
        for row in selected_rows
    ]


def class_distribution(records: list[dict]) -> list[dict]:
    counts = Counter(record["class_id"] for record in records)
    total = sum(counts.values())
    return [
        {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "objects": counts.get(class_id, 0),
            "percentage": round(100 * counts.get(class_id, 0) / total, 4) if total else 0,
        }
        for class_id in sorted(CLASS_NAMES)
    ]


def build_summary(rows: list[dict], image_records: list[dict], objects: list[dict], dimensions: Counter) -> dict:
    existing = sum(1 for row in rows if row["image_path"].exists())
    missing = len(rows) - existing
    objects_per_image = [len(row["objects"]) for row in rows]
    widths = [obj["width"] for obj in objects]
    heights = [obj["height"] for obj in objects]
    areas = [obj["area_px"] for obj in objects]
    angles = [obj["angle"] for obj in objects]
    dimension_rows = [
        {"width": width, "height": height, "images": count}
        for (width, height), count in sorted(dimensions.items())
    ]
    return {
        "images_in_csv": len(rows),
        "images_on_disk": existing,
        "missing_images": missing,
        "unique_clips": len({row["clip_id"] for row in rows}),
        "empty_frames": sum(1 for row in rows if not row["objects"]),
        "annotated_frames": sum(1 for row in rows if row["objects"]),
        "total_objects": len(objects),
        "objects_per_image": summarize_numbers(objects_per_image),
        "box_width_px": summarize_numbers(widths),
        "box_height_px": summarize_numbers(heights),
        "box_area_px": summarize_numbers(areas),
        "angle_deg": summarize_numbers(angles),
        "out_of_bounds_objects": sum(int(obj["out_of_bounds"]) for obj in objects),
        "image_dimensions": dimension_rows,
        "notes_for_paper": [
            "Splits are generated at clip level to reduce frame leakage between train and validation.",
            "Class imbalance should be reported because Macro AP penalizes rare-class failures.",
            "Sample sizes larger than available data are capped and marked in split manifests.",
        ],
    }


def write_markdown_summary(path: Path, summary: dict, distribution: list[dict]) -> None:
    lines = [
        "# Dataset audit summary",
        "",
        f"- Images in CSV: {summary['images_in_csv']}",
        f"- Images on disk: {summary['images_on_disk']}",
        f"- Missing images: {summary['missing_images']}",
        f"- Unique clips: {summary['unique_clips']}",
        f"- Empty frames: {summary['empty_frames']}",
        f"- Total annotated objects: {summary['total_objects']}",
        f"- Out-of-bounds objects: {summary['out_of_bounds_objects']}",
        "",
        "## Class distribution",
        "",
        "| Class | Objects | Percentage |",
        "| --- | ---: | ---: |",
    ]
    for row in distribution:
        lines.append(f"| {row['class_name']} | {row['objects']} | {row['percentage']:.2f}% |")
    lines.extend(
        [
            "",
            "## Paper-useful claims to verify",
            "",
            "- Dataset integrity can be reported using `dataset_summary.json`.",
            "- Class imbalance can motivate Macro AP and class-aware sampling.",
            "- Clip-level splitting should be used in all experiments.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path, img_dir)
    rows.sort(key=lambda row: row["id"])
    image_records, dimensions = inspect_images(rows)
    objects = object_records(rows)
    distribution = class_distribution(objects)
    summary = build_summary(rows, image_records, objects, dimensions)

    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(output_dir / "image_dimensions.csv", image_records, ["id", "width", "height"])
    write_csv(
        output_dir / "objects_long.csv",
        objects,
        [
            "image_id",
            "clip_id",
            "class_id",
            "class_name",
            "cx",
            "cy",
            "width",
            "height",
            "angle",
            "area_px",
            "area_ratio",
            "out_of_bounds",
        ],
    )
    write_csv(output_dir / "class_distribution.csv", distribution, ["class_id", "class_name", "objects", "percentage"])

    dimension_table = [
        {"width": width, "height": height, "images": count}
        for (width, height), count in sorted(dimensions.items())
    ]
    write_csv(output_dir / "dimension_distribution.csv", dimension_table, ["width", "height", "images"])

    for sample_size in args.sample_sizes:
        split = make_clip_level_split(rows, sample_size, args.val_ratio, args.seed)
        write_csv(
            output_dir / f"split_{sample_size}.csv",
            split,
            ["id", "clip_id", "split", "object_count", "available_subset_size", "requested_subset_size"],
        )

    write_markdown_summary(output_dir / "paper_dataset_summary.md", summary, distribution)
    print(json.dumps(summary, indent=2))
    print(f"Wrote dataset audit artifacts to {output_dir}")


if __name__ == "__main__":
    main()
