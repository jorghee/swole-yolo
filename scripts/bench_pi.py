#!/usr/bin/env python
"""Benchmark de modelos NCNN (Ultralytics) en Raspberry Pi, con metadato completo.

Reemplaza al script original, que se perdio. Mide por imagen (batch 1):
  - latencia de pared (lectura + preproceso + red + NMS)
  - desglose preproceso / inferencia / postproceso que reporta Ultralytics
y registra todo lo que el paper original no guardo: versiones de ncnn,
ultralytics y python, governor y frecuencia de CPU, hilos/afinidad,
temperatura y flag de throttling, y hash de los archivos del modelo.

Uso en la Pi:
    python scripts/bench_pi.py --model pi_models/m2_ncnn_model --imgsz 544 \
        --images-dir test_bank/ --n 300 --warmup 5 --out results/pi/m2.json
    # opcional: mAP en el dispositivo (equivalencia PyTorch vs NCNN)
    python scripts/bench_pi.py ... --eval-data yolo_ds/dataset_2500/data.yaml

Para limitar nucleos: --cores 0-3  (usa sched_setaffinity). Verifica en el JSON
(cpu_percent_per_core_mean) cuantos nucleos se usaron de verdad.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from importlib import metadata
from pathlib import Path


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def temp_c() -> float | None:
    raw = read("/sys/class/thermal/thermal_zone0/temp")
    return int(raw) / 1000 if raw else None


def throttled() -> str | None:
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return out.strip().split("=")[-1]
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_freq_mhz() -> float | None:
    raw = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return int(raw) / 1000 if raw else None


def pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sha256_dir(model_dir: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(model_dir.iterdir()) if p.suffix in {".param", ".bin", ".onnx", ".pt"}}


def environment(args) -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "ultralytics": pkg_version("ultralytics"),
        "ncnn": pkg_version("ncnn"),
        "numpy": pkg_version("numpy"),
        "opencv": pkg_version("opencv-python") or pkg_version("opencv-python-headless"),
        "cpu_model": next((l.split(":", 1)[1].strip() for l in (read("/proc/cpuinfo") or "").splitlines()
                           if l.startswith(("Model", "model name"))), None),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_freq_mhz_start": cpu_freq_mhz(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mem_total_kb": next((int(l.split()[1]) for l in (read("/proc/meminfo") or "").splitlines()
                              if l.startswith("MemTotal")), None),
        "args": {k: str(v) for k, v in vars(args).items()},
    }


def parse_cores(spec: str) -> set[int]:
    cores: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cores.update(range(int(lo), int(hi or lo) + 1))
    return cores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Carpeta *_ncnn_model")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--images-dir", required=True, help="Banco de imagenes (se ordenan por nombre)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cores", help="p.ej. 0-3; fija la afinidad de CPU")
    ap.add_argument("--sample-every", type=int, default=10, help="Cada cuantas imagenes se muestrean temp/CPU")
    ap.add_argument("--eval-data", help="data.yaml: si se da, calcula mAP en el dispositivo sobre split=test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import psutil
    from ultralytics import YOLO

    if args.cores:
        os.sched_setaffinity(0, parse_cores(args.cores))

    images = sorted(p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < args.n + args.warmup:
        raise SystemExit(f"Banco con {len(images)} imagenes; se necesitan {args.n + args.warmup}.")
    warm, bank = images[: args.warmup], images[args.warmup: args.warmup + args.n]

    model_dir = Path(args.model)
    model = YOLO(str(model_dir), task="obb")
    for path in warm:
        model.predict(str(path), imgsz=args.imgsz, device="cpu", verbose=False)

    psutil.cpu_percent(percpu=True)  # reinicia el contador
    lat, pre, inf, post, dets = [], [], [], [], 0
    samples = [{"i": 0, "temp_c": temp_c(), "freq_mhz": cpu_freq_mhz(), "throttled": throttled()}]
    cpu_samples: list[list[float]] = []
    t0 = time.perf_counter()
    for i, path in enumerate(bank, 1):
        start = time.perf_counter()
        res = model.predict(str(path), imgsz=args.imgsz, device="cpu", verbose=False)[0]
        lat.append((time.perf_counter() - start) * 1000)
        pre.append(res.speed["preprocess"]); inf.append(res.speed["inference"]); post.append(res.speed["postprocess"])
        dets += 0 if res.obb is None else len(res.obb)
        if i % args.sample_every == 0:
            cpu_samples.append(psutil.cpu_percent(percpu=True))
            samples.append({"i": i, "temp_c": temp_c(), "freq_mhz": cpu_freq_mhz(), "throttled": throttled()})
    wall = time.perf_counter() - t0

    lat_sorted = sorted(lat)
    per_core = [round(statistics.mean(c[j] for c in cpu_samples), 1) for j in range(len(cpu_samples[0]))] if cpu_samples else []
    result = {
        "environment": environment(args),
        "model_dir": str(model_dir),
        "model_files_sha256_16": sha256_dir(model_dir),
        "imgsz": args.imgsz,
        "images": len(bank),
        "warmup_images": len(warm),
        "latency_ms": {
            "mean": statistics.mean(lat), "median": statistics.median(lat), "std": statistics.pstdev(lat),
            "p95": lat_sorted[int(0.95 * (len(lat) - 1))], "min": lat_sorted[0], "max": lat_sorted[-1],
        },
        "ultralytics_speed_ms_mean": {"preprocess": statistics.mean(pre), "inference": statistics.mean(inf),
                                      "postprocess": statistics.mean(post)},
        "fps": 1000 / statistics.mean(lat),
        "wall_seconds_total": wall,
        "cpu_percent_per_core_mean": per_core,
        "cpu_percent_mean": round(statistics.mean(per_core), 1) if per_core else None,
        "thermal_samples": samples,
        "detections_per_image": dets / len(bank),
    }

    if args.eval_data:
        m = model.val(data=args.eval_data, split="test", imgsz=args.imgsz, device="cpu", verbose=False,
                      project=str(Path(args.out).parent / "val_runs"), exist_ok=True)
        result["on_device_eval"] = {k.replace("metrics/", "").replace("(B)", ""): float(v)
                                    for k, v in m.results_dict.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    lm = result["latency_ms"]
    print(f"{model_dir.name}: {lm['mean']:.1f} ms (p95 {lm['p95']:.1f}) -> {result['fps']:.2f} FPS; "
          f"governor={result['environment']['cpu_governor']} ncnn={result['environment']['ncnn']}")


if __name__ == "__main__":
    main()
