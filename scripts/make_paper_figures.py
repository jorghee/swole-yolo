#!/usr/bin/env python
"""Regenera las figuras cualitativas del paper sobre el split de TEST (no validacion).

Salidas en paper/:
  fig_confusion_test.png     matriz de confusion normalizada del modelo 70 ep / 640 px en test
  fig_qualitative_test.png   recorte con predicciones sobre una imagen de test
y results/paper_figures_test_metrics.json con las cifras que cita el texto
(AP@50 por clase y entradas de la matriz de confusion).

Uso:
    python scripts/make_paper_figures.py --yolo-data-dir /tmp/yolo_ds --device cpu
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swole_yolo import data  # noqa: E402
from swole_yolo.finetune import flatten_val_metrics  # noqa: E402

SIZE, IMGSZ, CKPT = 2500, 640, ROOT / "models" / "ds2500_70ep_640px.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-data-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    d = ROOT / "data" / f"dataset_{SIZE}"
    rows = data.load_rows(d / f"etiquetas_{SIZE}.csv", d / "images")
    tr, va, te = data.clip_level_indices(rows, 0.15, 0.15, 42 + SIZE)
    dest = Path(args.yolo_data_dir) / f"dataset_{SIZE}"
    yaml_path = dest / "data.yaml"
    if not yaml_path.exists():
        yaml_path = data.build_yolo_dataset(rows, {"train": tr, "val": va, "test": te}, dest)

    model = YOLO(str(CKPT))
    runs = Path(args.yolo_data_dir) / "val_runs"
    m = model.val(data=str(yaml_path), split="test", imgsz=IMGSZ, device=args.device, verbose=False,
                  plots=True, project=str(runs), name="fig_test", exist_ok=True)
    flat = flatten_val_metrics(m)

    # --- matriz de confusion (Ultralytics: filas = predicho, columnas = real, con fondo al final) ---
    mat = m.confusion_matrix.matrix.astype(float)
    colsum = mat.sum(axis=0, keepdims=True)
    norm = mat / np.maximum(colsum, 1e-9)
    bg, moto = len(data.CLASS_NAMES), data.CLASS_NAMES.index("motorcycle")
    stats = {
        "metrics": {k: v for k, v in flat.items() if k != "per_class_ap50"},
        "per_class_ap50": flat.get("per_class_ap50"),
        "motorcycle_true_assigned_to_background": float(norm[bg, moto]),
        "background_true_assigned_to_motorcycle": float(norm[moto, bg]),
        "motorcycle_true_correct": float(norm[moto, moto]),
    }
    shutil.copy(runs / "fig_test" / "confusion_matrix_normalized.png", ROOT / "paper" / "fig_confusion_test.png")
    # Recortar el margen blanco para que el texto sea legible a ancho de columna.
    from PIL import Image, ImageChops

    conf_png = ROOT / "paper" / "fig_confusion_test.png"
    im = Image.open(conf_png).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:
        pad = 12
        bbox = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0), min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height))
        im.crop(bbox).save(conf_png)

    # --- ejemplo cualitativo: imagen de test con moto y otro vehiculo no-auto; recorte 640x640 ---
    # Criterio: el modelo detecta una moto y al menos un vehiculo que no es auto ni moto
    # (primera imagen de test en orden de id que cumple; no se elige por estetica).
    moto_idx = data.CLASS_NAMES.index("motorcycle")
    car_idx = data.CLASS_NAMES.index("car")
    best = res = None
    for i in te:
        r = model.predict(str(rows[i]["image_path"]), imgsz=IMGSZ, device=args.device, verbose=False, conf=0.25)[0]
        cls = set(int(c) for c in r.obb.cls.tolist()) if r.obb is not None else set()
        if moto_idx in cls and cls - {moto_idx, car_idx} and 8 <= len(rows[i]["boxes"]) <= 25:
            best, res = rows[i], r
            break
    if best is None:
        raise SystemExit("No se encontro una imagen de test adecuada.")
    canvas = res.plot(line_width=2, font_size=14)
    k = [int(c) for c in res.obb.cls.tolist()].index(moto_idx)
    cx, cy = (float(v) for v in res.obb.xywhr[k][:2])
    h, w = canvas.shape[:2]
    x0 = int(min(max(cx - 320, 0), w - 640))
    y0 = int(min(max(cy - 320, 0), h - 640))
    cv2.imwrite(str(ROOT / "paper" / "fig_qualitative_test.png"), canvas[y0:y0 + 640, x0:x0 + 640])
    stats["qualitative_image"] = best["id"]
    stats["qualitative_crop_xy"] = [x0, y0]

    out = ROOT / "results" / "paper_figures_test_metrics.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
