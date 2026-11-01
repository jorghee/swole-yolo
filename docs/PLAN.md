# Plan de acción: revisión post-SIMBig (rama `dev-hot-fix`)

Contexto: el paper "Fine-Tuning YOLO11n-OBB for Oriented Vehicle Detection in Peruvian Traffic Scenes" fue rechazado en SIMBig 2026 (2 reseñas: weak reject y reject). El plazo del póster (15-sep-2026) ya pasó, así que el objetivo es una revista Q3 o superior.

> La rama se llama `dev-hot-fix` porque git no permite `dev/hot-fix` mientras exista la rama `dev`.

## Hallazgos de la auditoría inicial

Paper:
1. Existe un subconjunto de 3 000 imágenes entrenado (mAP@50 = 0.885, menor que el de 2 500) que el paper no reporta.
2. El escalado usa un test distinto por subconjunto: confunde "más datos" con "otro test". Sin semillas repetidas ni intervalos de confianza.
3. 544 px/50 épocas vs 640 px/70 épocas cambia dos factores a la vez.
4. El texto habla de vista "oblicua / cámara elevada", pero la Fig. 2 es cenital (dron). Verificar y alinear.
5. Figs. 2 y 3 salen del split de validación (usado para elegir el checkpoint), no del test.
6. El script del benchmark en la Raspberry Pi no está en el repo; faltan versión de NCNN, hilos y governor de CPU.
7. Sin baselines, sin comparación entre arquitecturas ni entre hardware; sin cuantificación (conteo).

Repo:
- Tres carpetas "pytorch" (`main/pytorch_implementation`, `main/pytorch-implementation`, `pytorch-implementation`), dos `particionar.py` distintos, `main/tensorflow_implementation` vacía.
- El script que generó los resultados es `finetune/messi.py`; su lanzador `messi.sh` llama a `finetune_yolo_obb.py`, que no existe.
- El README describe YOLOv8 y una GPU de 24 GB, no lo que se hizo.
- Git versiona `.pt`, `.onnx` y `__pycache__`; `.git` pesa ~209 MB. Restos de compilación LaTeX en `paper/`.
- `data/dataset_2500/images` tiene 2515 jpg para 2500 filas en el CSV (15 imágenes sobrantes por aclarar).

## Etapa 0: línea base reproducible
- [ ] Rama creada, versiones del entorno registradas (Ultralytics, torch, ncnn).
- [ ] Re-evaluar los checkpoints existentes y confirmar las Tablas 3-5.
- [ ] Recuperar el script del benchmark de la Raspberry Pi.
- Hecho cuando: un comando reproduce las tablas del paper.

## Etapa 1: ordenar el repo
- Estructura: `src/swole_yolo/` (convert, split, train, evaluate, bench_pi), `configs/`, `scripts/`, `results/`, `paper/`, `docs/`, `archive/`.
- Un solo proyecto `uv` con `pyproject.toml`; fusionar los `particionar.py`; renombrar `messi.py`.
- Sacar pesos y `__pycache__` de git (Releases o DVC). Reescribir el historial queda como decisión aparte (afecta a los 4 coautores).
- README nuevo con el flujo completo.
- Hecho cuando: la Etapa 0 da los mismos números con el código nuevo.

## Etapa 2: guía principiante a experto + docs en Zensical
Imágenes y redes → detección (IoU, NMS, anchor-free) → métricas (P, R, AP, mAP, matriz de confusión) → OBB (parametrización, ecuación 1, DOTA) → YOLO11 y transfer learning → datos (fuga entre frames, split por clip, desbalance) → edge (ONNX, NCNN, latencia vs throughput, P95, throttling, cuantización) → diseño de experimentos y lectura de papers. Cada tema enlaza al código donde aparece.

## Etapa 3: afinar el paper actual
- Reportar o justificar el subconjunto de 3 000.
- Figs. 2 y 3 sobre test; corregir descripción del punto de vista.
- Puntos del revisor 2: keywords en el abstract, párrafo de organización, qué es NCNN y por qué, URL del código, descripción de la arquitectura.
- Moderar el título. Opcional: preprint en arXiv.

## Etapa 4: revisión de literatura (últimos 3-5 años)
YOLO-OBB y detección rotada liviana; detección vehicular en Latinoamérica; YOLO en Raspberry Pi/Jetson/NCNN; estudios de data scaling; tracking y conteo (ByteTrack, BoT-SORT). Tabla por fuente: método, dataset, métricas, hardware, diferencia con nuestro trabajo. Elegir revistas objetivo verificando cuartil en SJR/Scimago.

## Etapa 5: plan de mejora para Q3+
- [M] Escalado con test fijo común, 5+ semillas, media e IC, curva hasta 3 000+.
- [M] Baselines con mismos datos y hardware: YOLOv8n-OBB, YOLO11s-OBB, HBB equivalente, entrenamiento desde cero.
- [M] Ablación ordenada resolución × épocas.
- [M] Benchmark en la Pi controlado (hilos, governor, versión NCNN), mAP en dispositivo, NCNN vs ONNX Runtime vs INT8.
- [S] Más de un hardware (Pi 4, Pi 5, Jetson o CPU de escritorio).
- [S] Tracking + conteo vehicular contra conteo real.
- [S] Análisis de errores (motocicleta, tamaño de objeto, noche/clima).
- [C] Prueba térmica y de energía de varias horas; publicar splits y pesos.
- Luego: redacción completa y carta de respuesta a los revisores.
