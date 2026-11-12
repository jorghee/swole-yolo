#!/usr/bin/env python
"""Fine-tuning de YOLO11n-obb para deteccion de vehiculos, con despliegue en Raspberry Pi.

Estrategia (decidida tras los experimentos desde-cero, que quedan como baseline
en archive/baseline_from_scratch):
- Se parte de yolo11n-obb.pt, preentrenado en DOTA (imagenes aereas con
  vehiculos y cajas orientadas).
- Decisiones metodologicas:
  * split por CLIPS completos (sin fuga de frames entre train/val/test),
  * misma semilla por tamano de dataset (seed + size),
  * el test se evalua UNA sola vez, con el mejor checkpoint.
- Para cada subset (1000..3000) se genera un dataset en formato YOLO-OBB,
  se fine-tunea, se evalua en test y se exporta el modelo a NCNN/ONNX.

Salidas por experimento:
- result_dataset_<n>.json  : metricas de test + latencia CPU + tamanos
- experiment_summary.csv / paper_experiment_summary.md
- pesos: best.pt + exportes .onnx y carpeta *_ncnn_model para la Pi

La conversion de anotaciones y el split viven en swole_yolo.data.

Uso:  python -m swole_yolo.finetune --data-root data --output-dir results/mi_run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from swole_yolo.data import (
    CLASS_NAMES,
    build_yolo_dataset,
    clip_level_indices,
    load_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data", help="Directorio con dataset_1000, ..., dataset_3000.")
    parser.add_argument("--yolo-data-dir", default="yolo_datasets", help="Donde se generan los datasets en formato YOLO-OBB.")
    parser.add_argument("--output-dir", default="results/finetune")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[1000, 1500, 2000, 2500, 3000])
    parser.add_argument("--model", default="yolo11n-obb.pt", help="Checkpoint preentrenado de Ultralytics (DOTA).")
    parser.add_argument("--epochs", type=int, default=50, help="Ultralytics ya trae early stopping (patience).")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=540, help="Ultralytics lo ajusta a multiplo de 32 (540 -> 544).")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="p.ej. '0' para GPU, 'cpu'. None = Ultralytics decide.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--benchmark-images", type=int, default=30, help="Imagenes de test para medir latencia CPU.")
    parser.add_argument("--benchmark-threads", type=int, default=4, help="Hilos del proxy CPU de escritorio.")
    parser.add_argument("--skip-convert", action="store_true", help="Reusar los datasets YOLO ya generados.")
    parser.add_argument("--export-formats", nargs="+", default=["onnx", "ncnn"],
                        help="Formatos de despliegue para la Pi.")
    return parser.parse_args()


def benchmark_cpu(weights: Path, image_paths: list[Path], img_size: int, threads: int) -> dict:
    """Latencia/FPS en CPU de escritorio con hilos limitados (solo proxy de desarrollo).

    Las cifras de despliegue se miden en la Pi real con scripts/bench_pi.py.
    """
    import torch
    from ultralytics import YOLO

    previous_threads = torch.get_num_threads()
    if threads > 0:
        torch.set_num_threads(threads)
    try:
        model = YOLO(str(weights))
        model.predict(str(image_paths[0]), imgsz=img_size, device="cpu", verbose=False)  # warmup
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
    try:
        per_class = {}
        box = metrics.box
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
    model = YOLO(args.model)
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

    # --- Benchmark CPU de escritorio (proxy) ---
    test_images = sorted((yolo_dataset / "images" / "test").iterdir())[: args.benchmark_images]
    inference = benchmark_cpu(best_weights, test_images, args.img_size, args.benchmark_threads)

    # --- Export para la Pi ---
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
        "Test split evaluated once with the best checkpoint; clip-level splits.",
        "CPU latency is a desktop proxy; deployment numbers are measured on-device with scripts/bench_pi.py.",
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
    print("Para la Raspberry: usa la carpeta *_ncnn_model con scripts/bench_pi.py.")


if __name__ == "__main__":
    main()
