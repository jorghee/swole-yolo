# Resultados y lectura crítica

Todas las cifras de esta página provienen de `results/` y fueron **verificadas** re-evaluando los checkpoints (ver [Reproducir](reproduccion.md)).

## Detección (offline, PyTorch)

| Config. | Train / Val / Test | mAP@50 | mAP@50–95 | Precisión | Recall | mAP@50 en **val** | test − val |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 000 · 50 ép · 544 | 700 / 150 / 150 | 0.662 | 0.548 | 0.774 | 0.642 | 0.704 | −0.042 |
| 1 500 · 50 ép · 544 | 1 050 / 225 / 225 | 0.817 | 0.677 | 0.835 | 0.706 | 0.824 | −0.007 |
| 2 000 · 50 ép · 544 | 1 400 / 300 / 300 | 0.813 | 0.687 | 0.862 | 0.722 | 0.873 | −0.060 |
| 2 500 · 50 ép · 544 | 1 748 / 376 / 376 | **0.914** | 0.785 | 0.905 | 0.852 | 0.860 | **+0.054** |
| 3 000 · 50 ép · 544 | 2 098 / 450 / 452 | 0.885 | 0.755 | 0.854 | 0.833 | 0.895 | −0.010 |
| 2 500 · 70 ép · 640 | 1 748 / 376 / 376 | **0.939** | **0.817** | 0.928 | 0.860 | 0.919 | +0.020 |

!!! warning "La fila de 3 000 no aparece en el paper"
    Se entrenó y evaluó (`results/finetune_50ep_544px/result_dataset_3000.json`), pero el paper corta la curva en 2 500. Dar un mAP@50 menor con más datos no es un error: es información que debe reportarse y explicarse.

## Por clase (modelo 640 px, test; Tabla 5 del paper)

| Clase | n test | AP@50 |
| --- | ---: | ---: |
| `car` | 3 515 | 0.993 |
| `truck` | 242 | 0.981 |
| `van` | 70 | 0.962 |
| `minibus` | 161 | 0.941 |
| `microbus` | 23 | 0.924 |
| `mototaxi` | 38 | 0.915 |
| `motorcycle` | 365 | **0.749** |
| `bus` | 20 | 0.995 |
| `articulated_bus` | 2 | 0.995 |

El mAP@50 (0.9394) es exactamente el promedio simple de estos 9 valores.

## Raspberry Pi 4 B (NCNN, batch 1, 300 imágenes) — medición original

| Modelo | Latencia media ± σ | P95 | FPS | Red | Temp. inicio→fin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 000 · 544 | 302.4 ± 5.3 ms | 311.5 | 3.31 | 253.6 ms | 48.2 → 65.2 °C |
| 2 500 · 544 | 300.5 ± 2.6 ms | 305.3 | **3.33** | 254.9 ms | 50.6 → 70.1 °C |
| 2 500 · 640 | 399.9 ± 3.6 ms | 405.9 | 2.50 | 353.4 ms | 49.7 → 70.1 °C |

!!! info "Estado de esta medición"
    El script que la produjo se perdió. Se reescribió como `scripts/bench_pi.py`, que registra además versiones, governor y la serie térmica. **Pendiente: volver a medir en la Pi** y comparar. Las cifras anteriores son plausibles (son internamente coherentes, ver el [capítulo 7](../guia/07-edge.md#73-como-medir-velocidad-correctamente)) pero no auditables.

## Lectura crítica: qué se puede y qué no se puede afirmar

### Se puede afirmar

- Un YOLO11n-OBB fine-tuneado alcanza mAP@50 ≈ 0.91–0.94 en un test de clips no vistos de este dataset (un solo split, sin intervalo).
- El modelo ocupa ≈ 10 MB en NCNN y corre a ≈ 2.5–3.3 FPS en una Pi 4 (medición no auditable).
- El tiempo de red escala con el número de píxeles (×1.386 vs ×1.384).

### No se puede afirmar (todavía)

| Afirmación tentadora | Por qué no |
| --- | --- |
| "Más datos mejoran el modelo de forma sistemática" | el 3 000 empeora; el test cambia con cada tamaño; los clips casi no aumentan; sin semillas |
| "2 500 imágenes son suficientes" | el máximo local de 2 500 puede ser ruido de test (+5.4 puntos sobre val) |
| "640 px es mejor que 544 px" | además de la resolución cambian las épocas, y el de 50 épocas no había convergido |
| "OBB es mejor que HBB aquí" | no se probó HBB; 81 % de las cajas están alineadas con los ejes |
| "Funciona en tiempo real" | el propio paper lo niega: 3.3 FPS < 10 FPS del video |
| "No hay throttling" | solo 90–120 s de prueba y la temperatura seguía subiendo |

### Debilidades no mencionadas por los revisores

1. **Fig. 2 y Fig. 3 usan validación**, no test.
2. **Motivación incorrecta**: el paper habla de vistas oblicuas y cámaras elevadas; las imágenes son **cenitales de dron**.
3. **Subconjuntos anidados con los mismos 1 088 clips** de 1 500 en adelante.
4. **Objetos diminutos a la resolución de entrada** (moto ≈ 3.6–4.3 px): explica `motorcycle` = 0.749 y sugiere que la resolución, no solo la cantidad de datos, es el factor crítico.
5. **Sin rotaciones en el entrenamiento** (`degrees=0`) en un detector orientado.
6. **Optimizador no declarado** (`optimizer=auto`).
7. **Tabla 3 (FPS de escritorio)** con 6 % de variación entre modelos idénticos: ruido, sin valor informativo.

Todas estas observaciones alimentan el [plan de revisión](../PLAN.md).
