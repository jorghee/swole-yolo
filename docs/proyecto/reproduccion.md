# Reproducir los resultados

## Requisitos

- Linux o macOS con Python ≥ 3.10 y [`uv`](https://docs.astral.sh/uv/).
- Las imágenes `data/dataset_*/images/*.jpg` (no están en git; hay que copiarlas desde la fuente original).
- Para entrenar: GPU recomendada. Para **solo evaluar**, basta una CPU (≈ 4 minutos para los 6 checkpoints).

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

!!! note "Versiones con las que se verificó"
    Ultralytics 8.4.211, PyTorch 2.15 (CPU), Python 3.12. El entrenamiento original corrió en un VPS con otra versión no registrada; por eso la reproducción es de **evaluación**, no de reentrenamiento bit a bit.

## 1. Verificar los números del paper (sin reentrenar)

```bash
python scripts/repro_baseline.py --yolo-data-dir /tmp/yolo_ds --device cpu
```

Reconstruye cada split por clips con la misma semilla, evalúa el checkpoint en test y compara con `result_dataset_N.json`. Esperado: diferencia absoluta máxima ≈ 0.0008 (ruido numérico CPU vs. GPU). Escribe `results/baseline_repro.json`.

Opciones: `--only 2500` limita a un tamaño; `--device 0` usa GPU.

## 2. Reentrenar

```bash
scripts/run_finetune.sh                         # 2 500 img, 70 ép, 640 px (defaults)
EPOCHS=50 IMG_SIZE=540 scripts/run_finetune.sh  # config de 50 épocas / 544 px
```

Las rutas se sobrescriben con `DATA_ROOT`, `OUTPUT_DIR`, `YOLO_DATA_DIR`. Para cambiar tamaños de dataset edita `DATASET_SIZES` en el script.

Directamente con Python:

```bash
PYTHONPATH=src python -m swole_yolo.finetune --data-root data \
    --output-dir results/mi_run --dataset-sizes 2500 --epochs 70 --img-size 640
```

!!! warning "No esperes los mismos decimales"
    Reentrenar con otra versión de Ultralytics, otra GPU o distinto orden de operaciones da métricas ligeramente distintas. Eso es normal y es justo el motivo por el que en el plan de mejora se piden varias semillas.

## 3. Exportar a NCNN

```bash
python scripts/export_ncnn.py models/ds2500_70ep_640px.pt --imgsz 640
# → models/_local/ds2500_70ep_640px/best_ncnn_model/
```

## 4. Benchmark en la Raspberry Pi

En la Pi (64-bit, Python ≥ 3.10) instala `ultralytics ncnn psutil`. Copia `scripts/bench_pi.py`, la carpeta `best_ncnn_model` y un banco de ≥ 305 imágenes de test.

```bash
# Fijar governor para reproducibilidad (requiere sudo)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

python scripts/bench_pi.py --model best_ncnn_model --imgsz 640 \
    --images-dir test_bank/ --n 300 --warmup 5 --out results/pi/m3_640.json
```

El JSON resultante incluye versiones (`ncnn`, `ultralytics`), governor, frecuencia, afinidad de CPU, uso por núcleo, serie de temperatura/flags de throttling y hashes de los archivos del modelo. Con `--eval-data <data.yaml>` calcula además el mAP **en el dispositivo**.

Checklist antes de medir: ventilación fija, Pi en reposo ~ 15 min, misma fuente de alimentación, sin procesos en segundo plano, repetir ≥ 3 veces y reportar la dispersión.

## 5. Construir esta documentación

```bash
uv pip install zensical
zensical serve      # vista previa en http://localhost:8000
zensical build      # genera site/
```

## Lagunas conocidas

| Laguna | Efecto |
| --- | --- |
| `train.csv` completo no disponible | no se pueden regenerar los subsets con `partition_dataset.py` |
| Sin log del entrenamiento original | optimizador `auto` sin confirmar |
| Script de la Pi original perdido | cifras de `pi_original_lost_script/` no auditables |
| 15 imágenes de más en `dataset_2500/images` | sin efecto (manda el CSV); origen por aclarar |
