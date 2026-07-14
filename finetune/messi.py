#!/usr/bin/env python
"""Fine-tuning de YOLO11n-obb para deteccion de vehiculos, con despliegue en Raspberry Pi.

Estrategia (decidida tras los experimentos desde-cero, que quedan como baseline):
- Se parte de yolo11n-obb.pt, preentrenado en DOTA (imagenes aereas con
  vehiculos y cajas orientadas): el transfer es casi ideal para trafico
  visto desde camara elevada/dron.
- Se conservan las decisiones metodologicas del pipeline anterior:
  * split por CLIPS completos (sin fuga de frames entre train/val/test),
  * misma semilla por tamano de dataset (seed + size) para comparabilidad,
  * el test se evalua UNA sola vez, con el mejor checkpoint.
- Para cada subset (1000..3000) se genera un dataset en formato YOLO-OBB,
  se fine-tunea, se evalua en test y se exporta el modelo a NCNN/ONNX
  (los formatos rapidos para ARM/Raspberry).

Salidas por experimento (paper-ready, como antes):
- result_dataset_<n>.json  : metricas de test + latencia CPU + tamanos
- experiment_summary.csv / paper_experiment_summary.md
- comparacion grafica de mAP50/precision/recall por tamano de dataset
- pesos: best.pt + exportes .onnx y carpeta *_ncnn_model para la Pi

Formato de anotacion de entrada (CSV):  Id, Target
  Target = "cls cx cy w h angulo; cls cx cy w h angulo; ..."  (pixeles, grados)
Formato de salida (YOLO OBB): por imagen un .txt con lineas
  cls_idx x1 y1 x2 y2 x3 y3 x4 y4   (esquinas normalizadas 0-1)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = [
    "car", "van", "microbus", "minibus", "bus",
    "articulated_bus", "truck", "mototaxi", "motorcycle",
]
NUM_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default="data", help="Directorio con dataset_1000, ..., dataset_3000.")
    parser.add_argument("--yolo-data-dir", default="yolo_datasets", help="Donde se generan los datasets en formato YOLO-OBB.")
    parser.add_argument("--output-dir", default="reports/paper_metrics/finetune")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[1000, 1500, 2000, 2500, 3000])
    parser.add_argument("--model", default="yolo11n-obb.pt", help="Checkpoint preentrenado de Ultralytics (DOTA).")
    parser.add_argument("--epochs", type=int, default=50, help="Ultralytics ya trae early stopping (patience).")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=540, help="540 es el nativo de YOLO; en la Pi se puede inferir a menos.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="p.ej. '0' para GPU, 'cpu'. None = Ultralytics decide.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--benchmark-images", type=int, default=30, help="Imagenes de test para medir latencia CPU.")
    parser.add_argument("--benchmark-threads", type=int, default=4, help="4 hilos aproxima una Raspberry Pi 4/5.")
    parser.add_argument("--skip-convert", action="store_true", help="Reusar los datasets YOLO ya generados.")
    parser.add_argument("--export-formats", nargs="+", default=["onnx", "ncnn"],
                        help="Formatos de despliegue para la Pi. NCNN es el mas rapido en ARM.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Conversion de anotaciones: CSV (cx,cy,w,h,angulo) -> YOLO OBB (4 esquinas)
# ---------------------------------------------------------------------------

def parse_target(target_text: str | None) -> list[dict[str, float]]:
    """Mismo parser del pipeline anterior: valida y normaliza cada caja."""
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
    """Split por clips completos (misma logica y semillas que el pipeline baseline).

    Aunque las imagenes esten curadas, frames del mismo clip comparten camara,
    escena e iluminacion; separar clips completos elimina esa fuga residual y
    es un punto metodologico defendible en el paper.
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
    """(cx,cy,w,h,angulo) -> 4 esquinas normalizadas [x1,y1,...,x4,y4].

    Es la misma geometria de oriented_box_points del pipeline anterior.
    Las coordenadas se acotan a [0,1] porque YOLO exige etiquetas dentro
    de la imagen (cajas parcialmente fuera se recortan al borde).
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
            # Symlink en vez de copia: los subsets comparten imagenes y el
            # disco del VPS es limitado.
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


# ---------------------------------------------------------------------------
# Fine-tuning, evaluacion en test y benchmark CPU
# ---------------------------------------------------------------------------

def benchmark_cpu(weights: Path, image_paths: list[Path], img_size: int, threads: int) -> dict:
    """Latencia/FPS en CPU con hilos limitados (proxy de la Raspberry).

    Se mide la prediccion completa de Ultralytics (preproceso + red + NMS)
    imagen por imagen (batch 1, el uso real en la Pi). Sigue siendo un proxy:
    la cifra definitiva del paper debe medirse en la Pi real con el export NCNN.
    """
    import torch
    from ultralytics import YOLO

    previous_threads = torch.get_num_threads()
    if threads > 0:
        torch.set_num_threads(threads)
    try:
        model = YOLO(str(weights))
        # Warmup (carga de pesos, primeras asignaciones de memoria).
        model.predict(str(image_paths[0]), imgsz=img_size, device="cpu", verbose=False)
        times = []
        for path in image_paths:
            start = time.perf_counter()
            model.predict(str(path), imgsz=img_size, device="cpu", verbose=False)
            times.append(time.perf_counter() - start)
    finally:
        torch.set_num_threads(previous_threads)
    mean_latency = sum(times) / max(len(times), 1)
    return {
        "latency_ms_per_image_cpu": mean_latency * 1000,
        "fps_cpu": 1 / mean_latency if mean_latency > 0 else 0,
        "benchmark_images": len(times),
        "benchmark_threads": threads if threads > 0 else previous_threads,
    }


def flatten_val_metrics(metrics) -> dict:
    """Extrae metricas del objeto de validacion de Ultralytics de forma robusta."""
    flat = {}
    results_dict = getattr(metrics, "results_dict", None) or {}
    for key, value in results_dict.items():
        clean = key.replace("metrics/", "").replace("(B)", "")
        try:
            flat[clean] = float(value)
        except (TypeError, ValueError):
            pass
    # AP50 por clase si esta disponible (util para la tabla por clase del paper).
    try:
        per_class = {}
        box = metrics.box  # en OBB este objeto expone ap50 y ap_class_index
        for class_index, ap50 in zip(box.ap_class_index.tolist(), box.ap50.tolist()):
            per_class[CLASS_NAMES[int(class_index)]] = float(ap50)
        flat["per_class_ap50"] = per_class
    except Exception:
        pass
    return flat


def experiment(args, dataset_size: int, output_dir: Path) -> dict:
    from ultralytics import YOLO

    dataset_dir = Path(args.data_root) / f"dataset_{dataset_size}"
    csv_path = dataset_dir / f"etiquetas_{dataset_size}.csv"
    img_dir = dataset_dir / "images"
    if not csv_path.exists() or not img_dir.exists():
        raise FileNotFoundError(f"Falta dataset_{dataset_size}: se esperaba {csv_path} y {img_dir}")
    rows = load_rows(csv_path, img_dir)
    if not rows:
        raise ValueError(f"dataset_{dataset_size} sin filas validas")

    # Misma semilla por tamano que el pipeline baseline: splits comparables.
    experiment_seed = args.seed + dataset_size
    train_idx, val_idx, test_idx = clip_level_indices(rows, args.val_ratio, args.test_ratio, experiment_seed)

    yolo_dataset = Path(args.yolo_data_dir) / f"dataset_{dataset_size}"
    yaml_path = yolo_dataset / "data.yaml"
    if not args.skip_convert or not yaml_path.exists():
        print(f"Convirtiendo dataset_{dataset_size} a formato YOLO-OBB "
              f"(train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})...")
        yaml_path = build_yolo_dataset(
            rows, {"train": train_idx, "val": val_idx, "test": test_idx}, yolo_dataset
        )

    # --- Fine-tuning ---
    model = YOLO(args.model)  # descarga el checkpoint preentrenado si no existe
    train_start = time.perf_counter()
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        workers=args.workers,
        seed=experiment_seed,
        project=str(output_dir / "runs"),
        name=f"dataset_{dataset_size}",
        exist_ok=True,
        verbose=False,
    )
    total_train_seconds = time.perf_counter() - train_start
    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"

    # --- Evaluacion: el TEST se toca una unica vez, con el mejor checkpoint ---
    best_model = YOLO(str(best_weights))
    test_metrics = flatten_val_metrics(
        best_model.val(data=str(yaml_path), split="test", imgsz=args.img_size, device=args.device, verbose=False)
    )

    # --- Benchmark CPU (proxy Raspberry) ---
    test_images = sorted((yolo_dataset / "images" / "test").iterdir())[: args.benchmark_images]
    inference = benchmark_cpu(best_weights, test_images, args.img_size, args.benchmark_threads)

    # --- Export para la Pi (NCNN es el formato rapido en ARM) ---
    exports = {}
    for fmt in args.export_formats:
        try:
            exported = best_model.export(format=fmt, imgsz=args.img_size)
            exports[fmt] = str(exported)
        except Exception as error:  # el export nunca debe tumbar el experimento
            print(f"Aviso: fallo el export a {fmt} ({error}).")
            exports[fmt] = None

    parameters = sum(p.numel() for p in best_model.model.parameters())
    result = {
        "dataset_size": dataset_size,
        "available_images": len(rows),
        "train_images": len(train_idx),
        "val_images": len(val_idx),
        "test_images": len(test_idx),
        "base_model": args.model,
        "epochs_requested": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "parameters": parameters,
        "model_size_mb": best_weights.stat().st_size / (1024 * 1024),
        "total_train_seconds": total_train_seconds,
        **{f"test_{key}": value for key, value in test_metrics.items() if key != "per_class_ap50"},
        "per_class_ap50": test_metrics.get("per_class_ap50", {}),
        **inference,
        "weights": str(best_weights),
        "exports": exports,
        "run_dir": str(run_dir),
    }
    (output_dir / f"result_dataset_{dataset_size}.json").write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Reportes agregados para el paper
# ---------------------------------------------------------------------------

def write_summary(output_dir: Path, results: list[dict]) -> None:
    fields = [
        "dataset_size", "available_images", "train_images", "val_images", "test_images",
        "parameters", "model_size_mb", "total_train_seconds",
        "test_mAP50", "test_mAP50-95", "test_precision", "test_recall",
        "latency_ms_per_image_cpu", "fps_cpu", "benchmark_threads",
    ]
    with (output_dir / "experiment_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# Fine-tuning summary (YOLO11n-obb, pretrained on DOTA)",
        "",
        "| Dataset | Train | Val | Test | Test mAP50 | Test mAP50-95 | Precision | Recall | Latency ms/img CPU | FPS CPU | Train s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['dataset_size']} | {row['train_images']} | {row['val_images']} | {row['test_images']} | "
            f"{row.get('test_mAP50', float('nan')):.4f} | {row.get('test_mAP50-95', float('nan')):.4f} | "
            f"{row.get('test_precision', float('nan')):.4f} | {row.get('test_recall', float('nan')):.4f} | "
            f"{row['latency_ms_per_image_cpu']:.2f} | {row['fps_cpu']:.2f} | {row['total_train_seconds']:.0f} |"
        )
    lines += [
        "",
        "Test split evaluated once with the best checkpoint; clip-level splits shared with the from-scratch baseline.",
        "CPU latency measured with limited threads as a Raspberry Pi proxy; final numbers must be measured on-device (NCNN export).",
    ]
    (output_dir / "paper_experiment_summary.md").write_text("\n".join(lines) + "\n")


def plot_comparison(output_dir: Path, results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [str(row["dataset_size"]) for row in results]
    x = list(range(len(sizes)))
    width = 0.36
    plt.figure(figsize=(7, 4))
    plt.bar([i - width / 2 for i in x], [row.get("test_mAP50", 0) for row in results], width, label="Test mAP@0.50")
    plt.bar([i + width / 2 for i in x], [row.get("test_mAP50-95", 0) for row in results], width, label="Test mAP@0.50-0.95")
    plt.xticks(x, [f"dataset_{size}" for size in sizes])
    plt.ylim(0, 1)
    plt.ylabel("Held-out test score")
    plt.title("YOLO11n-obb fine-tuning by dataset size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_metric_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    results = []
    for dataset_size in args.dataset_sizes:
        print(f"\n=== Fine-tuning dataset_{dataset_size} ===")
        results.append(experiment(args, dataset_size, output_dir))
    write_summary(output_dir, results)
    plot_comparison(output_dir, results)
    print(f"\nArtefactos escritos en {output_dir}")
    print("Para la Raspberry: usa el export NCNN (carpeta *_ncnn_model) con `YOLO('<carpeta>')` o ncnn nativo.")


if __name__ == "__main__":
    main()