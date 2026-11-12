#!/usr/bin/env python
"""Figura de tamano de datos (Fig. 1 del paper): mAP en test y mAP@50 en validacion vs imagenes de train.

Lee results/finetune_50ep_544px/ (experiment_summary.csv y runs/*/results.csv).
El mAP@50 de validacion es el de la epoca del checkpoint seleccionado
(maximo de fitness = 0.1*mAP50 + 0.9*mAP50-95, como hace Ultralytics).

Salida: paper/fig_scaling.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "finetune_50ep_544px"


def best_val_map50(run_csv: Path) -> float:
    with run_csv.open() as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    fit = [0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]) for r in rows]
    return float(rows[fit.index(max(fit))]["metrics/mAP50(B)"])


def main() -> None:
    with (RES / "experiment_summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    sizes = [int(r["dataset_size"]) for r in summary]
    t50 = [float(r["test_mAP50"]) for r in summary]
    t5095 = [float(r["test_mAP50-95"]) for r in summary]
    v50 = [best_val_map50(RES / "runs" / f"dataset_{n}" / "results.csv") for n in sizes]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(sizes, t50, "o-", color="#1f4e8c", label="Test mAP@50")
    ax.plot(sizes, t5095, "s--", color="#b5451b", label="Test mAP@50$-$95")
    ax.plot(sizes, v50, "^:", color="#7a7a7a", label="Validation mAP@50 (selected epoch)")
    for x, y in zip(sizes, t50):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    for x, y in zip(sizes, t5095):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
    ax.set_xticks(sizes)
    ax.set_xlabel("Training images")
    ax.set_ylabel("Score")
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.set_title("YOLO11n-OBB: 50 epochs, 544 px effective input", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ROOT / "paper" / "fig_scaling.pdf"
    fig.savefig(out)
    print(out)
    for n, a, b, c in zip(sizes, t50, t5095, v50):
        print(n, f"test50={a:.4f} test5095={b:.4f} val50={c:.4f}")


if __name__ == "__main__":
    main()
