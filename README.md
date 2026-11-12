# swole-yolo

Fine-tuning de **YOLO11n-OBB** (cajas orientadas) para detectar 9 categorías de vehículos en escenas de tráfico peruano, con benchmark de inferencia en **Raspberry Pi 4** (NCNN). Datos: [MTC SMART Challenge: IA para la Movilidad del Perú](https://www.kaggle.com/competitions/mtc-smart-challenge-ia-para-la-movilidad-del-peru) (Kaggle).

Paper asociado (`paper/`): *Fine-Tuning YOLO11n-OBB for Oriented Vehicle Detection in Peruvian Traffic Scenes: Data Scaling and Edge Benchmarking*. Fue rechazado en SIMBig 2026; el trabajo de revisión está planificado en [docs/PLAN.md](docs/PLAN.md).

## Resultados reportados en el paper

| Configuración | mAP@50 | mAP@50–95 | Carpeta de resultados |
| --- | ---: | ---: | --- |
| 1 000 img · 50 ep · 544 px | 0.662 | 0.548 | `results/finetune_50ep_544px/` |
| 1 500 img · 50 ep · 544 px | 0.817 | 0.677 | idem |
| 2 000 img · 50 ep · 544 px | 0.813 | 0.687 | idem |
| 2 500 img · 50 ep · 544 px | 0.914 | 0.785 | idem |
| 3 000 img · 50 ep · 544 px | 0.885 | 0.755 | idem (**no está en el paper**) |
| 2 500 img · 70 ep · 640 px | 0.939 | 0.817 | `results/finetune_70ep_640px/` |

Cada tamaño usa su propio split de test (por clips). Latencia en Pi 4 (NCNN): 300 ms (544 px) y 400 ms (640 px); ver `results/pi_original_lost_script/`.

## Estructura

```
src/swole_yolo/
  data.py          parseo de anotaciones, split por clips, conversión a YOLO-OBB (ec. 1 del paper)
  finetune.py      entrenamiento, evaluación en test, export a ONNX/NCNN
scripts/
  repro_baseline.py   re-evalúa los 6 checkpoints y compara con los números guardados
  bench_pi.py         benchmark NCNN en Raspberry Pi (con metadatos completos)
  export_ncnn.py      regenera NCNN/ONNX desde models/*.pt
  run_finetune.sh     instala dependencias y lanza el entrenamiento en un VPS
  partition_dataset.py   genera los subconjuntos 1000…3000 desde el train.csv completo
models/            los 6 checkpoints .pt del paper (derivados en models/_local/, no versionados)
results/           métricas por experimento (JSON/CSV), reproducción y benchmark original de la Pi
data/              CSV de etiquetas por subconjunto (las imágenes .jpg no se versionan)
paper/             LaTeX del paper
docs/              plan de revisión y documentación
archive/           lo exploratorio previo: baseline desde cero, intento previo en Windows, notebooks
```

## Puesta en marcha

```bash
uv venv && uv pip install -e .
```

Reproducir las métricas de test del paper sin reentrenar (necesita las imágenes en `data/dataset_*/images/`):

```bash
python scripts/repro_baseline.py --yolo-data-dir /tmp/yolo_ds --device cpu
```

Reentrenar (solo el tamaño 2 500, 70 épocas, 640 px por defecto):

```bash
scripts/run_finetune.sh
```

## Notas importantes

- **Código del paper = `src/swole_yolo/finetune.py`.** El baseline desde cero (red propia en PyTorch) está en `archive/baseline_from_scratch/` y **no** produjo los resultados del paper.
- El script original del benchmark de la Pi se perdió; `scripts/bench_pi.py` lo reemplaza. Las cifras de `results/pi_original_lost_script/` no son auditables hasta volver a medir.
- `data/dataset_2500/images/` contiene 15 imágenes más que filas tiene su CSV; el CSV es la fuente de verdad.
- Reproducción verificada con Ultralytics 8.4.211 / torch 2.15 (CPU); diferencia máxima 0.0008 en mAP frente a los valores guardados.
