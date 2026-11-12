#!/usr/bin/env python
"""Etapa 0: re-evalua los checkpoints existentes sobre su split de test y
compara contra los numeros guardados en result_dataset_<n>.json.

No entrena nada. Reutiliza la conversion y el split por clips de
swole_yolo.data para reconstruir exactamente el mismo test.

Uso:
    python scripts/repro_baseline.py --yolo-data-dir /tmp/yolo_ds --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swole_yolo import data  # noqa: E402
from swole_yolo.finetune import flatten_val_metrics  # noqa: E402

# (tamano, imgsz, checkpoint, json con los numeros guardados)
E50, E70 = "results/finetune_50ep_544px", "results/finetune_70ep_640px"
RUNS = [
    (1000, 540, "models/ds1000_50ep_544px.pt", f"{E50}/result_dataset_1000.json"),
    (1500, 540, "models/ds1500_50ep_544px.pt", f"{E50}/result_dataset_1500.json"),
    (2000, 540, "models/ds2000_50ep_544px.pt", f"{E50}/result_dataset_2000.json"),
    (2500, 540, "models/ds2500_50ep_544px.pt", f"{E50}/result_dataset_2500.json"),
    (3000, 540, "models/ds3000_50ep_544px.pt", f"{E50}/result_dataset_3000.json"),
    (2500, 640, "models/ds2500_70ep_640px.pt", f"{E70}/result_dataset_2500.json"),
]
KEYS = ["mAP50", "mAP50-95", "precision", "recall"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-data-dir", required=True, help="Donde se generan los datasets YOLO (fuera del repo).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", type=int, nargs="*", help="Limitar a estos tamanos de dataset.")
    args = ap.parse_args()

    from ultralytics import YOLO

    report = []
    for size, imgsz, ckpt, saved_json in RUNS:
        if args.only and size not in args.only:
            continue
        saved = json.loads((ROOT / saved_json).read_text())
        d = ROOT / "data" / f"dataset_{size}"
        rows = data.load_rows(d / f"etiquetas_{size}.csv", d / "images")
        tr, va, te = data.clip_level_indices(rows, 0.15, 0.15, 42 + size)
        dest = Path(args.yolo_data_dir) / f"dataset_{size}"
        yaml_path = dest / "data.yaml"
        if not yaml_path.exists():
            yaml_path = data.build_yolo_dataset(rows, {"train": tr, "val": va, "test": te}, dest)
        m = flatten_val_metrics(
            YOLO(str(ROOT / ckpt)).val(data=str(yaml_path), split="test", imgsz=imgsz, device=args.device, verbose=False,
                                  project=str(Path(args.yolo_data_dir) / "val_runs"), exist_ok=True)
        )
        row = {"size": size, "imgsz": imgsz, "n_test": len(te), "n_test_saved": saved["test_images"]}
        for k in KEYS:
            row[k] = m[k]
            row[f"{k}_saved"] = saved[f"test_{k}"]
            row[f"{k}_diff"] = m[k] - saved[f"test_{k}"]
        report.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    out = ROOT / "results" / "baseline_repro.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    worst = max((abs(r[f"{k}_diff"]) for r in report for k in KEYS), default=0.0)
    print(f"\nDiferencia absoluta maxima vs numeros guardados: {worst:.5f}")


if __name__ == "__main__":
    main()
