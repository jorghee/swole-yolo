#!/usr/bin/env python
"""Regenera los exportes NCNN/ONNX de un checkpoint de models/ (no se versionan: son derivados).

Uso:
    python scripts/export_ncnn.py models/ds2500_70ep_640px.pt --imgsz 640
    -> models/_local/ds2500_70ep_640px/best_ncnn_model/  (copiar a la Pi)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--formats", nargs="+", default=["ncnn", "onnx"])
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = args.checkpoint.parent / "_local" / args.checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        # Ultralytics exporta junto al .pt; lo movemos a la carpeta de derivados.
        exported = Path(YOLO(str(args.checkpoint)).export(format=fmt, imgsz=args.imgsz))
        target = out_dir / ("best_ncnn_model" if fmt == "ncnn" else f"best{exported.suffix}")
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(exported), str(target))
        print(f"{fmt}: {target}")


if __name__ == "__main__":
    main()
