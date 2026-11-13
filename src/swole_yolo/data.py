"""Datos: parseo de anotaciones, split por clips y conversion a YOLO-OBB.

Formato de entrada (CSV):  Id, Target
  Target = "cls cx cy w h angulo; cls cx cy w h angulo; ..."  (pixeles, grados)
  Id = "<clip_id>_<frame>"; las clases van de 1 a 9.
Formato de salida (YOLO OBB): por imagen un .txt con lineas
  cls_idx x1 y1 x2 y2 x3 y3 x4 y4   (esquinas normalizadas 0-1, cls_idx de 0 a 8)
"""

from __future__ import annotations

import csv
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = [
    "car", "van", "microbus", "minibus", "bus",
    "articulated_bus", "truck", "mototaxi", "motorcycle",
]
NUM_CLASSES = len(CLASS_NAMES)


def parse_target(target_text: str | None) -> list[dict[str, float]]:
    """Valida y normaliza cada caja de la columna Target."""
    if target_text is None or target_text.strip().lower() == "none":
        return []
    objects = []
    for raw_object in target_text.split(";"):
        parts = raw_object.strip().split()
        if not parts:
            continue
        if len(parts) != 6:
            raise ValueError(f"Anotacion invalida: {raw_object}")
        class_id = int(float(parts[0]))
        if not 1 <= class_id <= NUM_CLASSES:
            raise ValueError(f"Clase desconocida {class_id} en anotacion: {raw_object}")
        cx, cy, width, height, angle = map(float, parts[1:])
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensiones no positivas en anotacion: {raw_object}")
        objects.append({"class_id": class_id, "cx": cx, "cy": cy, "w": width, "h": height, "angle": angle % 360.0})
    return objects


def load_rows(csv_path: Path, img_dir: Path) -> list[dict]:
    rows = []
    missing = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            image_id = row["Id"]
            image_path = img_dir / f"{image_id}.jpg"
            if not image_path.exists():
                missing.append(image_path)
                continue
            rows.append({
                "id": image_id,
                "clip_id": image_id.rsplit("_", 1)[0],
                "image_path": image_path,
                "boxes": parse_target(row["Target"]),
            })
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} imagenes del CSV; primera: {missing[0]}")
    rows.sort(key=lambda r: r["id"])
    return rows


def clip_level_indices(rows: list[dict], val_ratio: float, test_ratio: float, seed: int):
    """Split por clips completos (sin fuga de frames entre train/val/test).

    Frames del mismo clip comparten camara, escena e iluminacion; separar clips
    completos elimina esa fuga. Los clips se barajan con `random.Random(seed)` y
    se asignan en orden: primero val, luego test, el resto a train.
    """
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("Ratios de val/test deben ser positivos y sumar menos de uno.")
    clip_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        clip_to_indices[row["clip_id"]].append(idx)
    clips = list(clip_to_indices)
    random.Random(seed).shuffle(clips)
    val_target = max(1, round(len(rows) * val_ratio))
    test_target = max(1, round(len(rows) * test_ratio))
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for clip in clips:
        if len(val_idx) < val_target:
            bucket = val_idx
        elif len(test_idx) < test_target:
            bucket = test_idx
        else:
            bucket = train_idx
        bucket.extend(clip_to_indices[clip])
    return train_idx, val_idx, test_idx


def obb_corners_normalized(box: dict[str, float], img_w: int, img_h: int) -> list[float]:
    """(cx,cy,w,h,angulo) -> 4 esquinas normalizadas [x1,y1,...,x4,y4]  (ecuacion 1 del paper).

    Las coordenadas se acotan a [0,1] porque YOLO exige etiquetas dentro de la
    imagen (cajas parcialmente fuera se recortan al borde).
    """
    angle = math.radians(box["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    half_w, half_h = box["w"] / 2, box["h"] / 2
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    flat: list[float] = []
    for dx, dy in corners:
        x = box["cx"] + dx * cos_a - dy * sin_a
        y = box["cy"] + dx * sin_a + dy * cos_a
        flat.append(min(max(x / img_w, 0.0), 1.0))
        flat.append(min(max(y / img_h, 0.0), 1.0))
    return flat


def build_yolo_dataset(rows: list[dict], indices_by_split: dict[str, list[int]], dest: Path) -> Path:
    """Crea images/labels por split (symlinks para no duplicar disco) y el data.yaml."""
    from PIL import Image

    if dest.exists():
        shutil.rmtree(dest)
    for split, indices in indices_by_split.items():
        img_out = dest / "images" / split
        lbl_out = dest / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            row = rows[idx]
            link = img_out / row["image_path"].name
            link.symlink_to(row["image_path"].resolve())
            with Image.open(row["image_path"]) as image:
                img_w, img_h = image.size
            lines = []
            for box in row["boxes"]:
                coords = obb_corners_normalized(box, img_w, img_h)
                lines.append(f"{box['class_id'] - 1} " + " ".join(f"{value:.6f}" for value in coords))
            (lbl_out / f"{row['id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml_path = dest / "data.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {dest.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )
    return yaml_path
