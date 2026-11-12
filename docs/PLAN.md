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

Hallazgos añadidos durante la Etapa 2 (medidos sobre los datos y checkpoints reales):

8. **Subconjuntos anidados** (1000 ⊂ 1500 ⊂ … ⊂ 3000) y, de 1500 en adelante, con los **mismos 1 088 clips**: "escalar datos" solo añade frames de las mismas escenas.
9. **Los modelos de 50 épocas no habían convergido**: mejor checkpoint en las épocas 43–50 (y 60 de 70). Agrava el confusor 544/50 vs 640/70.
10. **Ruido de test cuantificado:** mismo modelo, mAP@50 val 0.860 vs test 0.914 (+5.4 pts) en el subconjunto de 2 500; diferencias test − val entre −6.0 y +5.4 pts.
11. **61 % de las cajas tienen ángulo 0° y 81 % están a ≤ 2° de un eje**; solo 6.9 % se desvía más de 10°. El valor de OBB frente a HBB no está demostrado.
12. **Objetos diminutos:** a 544 px un auto mide ≈ 10 px y una moto ≈ 3.6 px de lado corto; explica `motorcycle` = 0.749.
13. Las imágenes son **vistas cenitales (nadir) de dron**; el paper dice "oblicuas / cámaras elevadas".
14. `degrees=0` (sin rotación como aumento), `optimizer=auto` sin confirmar, sin log de entrenamiento.
15. La Tabla 3 (FPS de escritorio) varía 6 % entre modelos idénticos: es ruido; responde la pregunta del revisor 2 ("¿por qué varía el FPS?").
16. **Convención de etiquetado:** en los ejemplos inspeccionados, vehículos estacionados fuera de la calzada no están anotados, y el modelo tampoco los predice (aprendió la convención). Falta documentar la política de anotación del dataset.
17. **El subconjunto de 3 000 se excluyó deliberadamente** (`paper/paper_guideline.md`: "su distribución está desproporcionada"). Medido: sus proporciones por clase son prácticamente idénticas a las de los otros tamaños (auto ≈ 80 %, moto ≈ 8 % en train/val/test). La razón documentada no se sostiene con los datos; ver "Decisión pendiente" en la Etapa 3.

Repo:
- Tres carpetas "pytorch" (`main/pytorch_implementation`, `main/pytorch-implementation`, `pytorch-implementation`), dos `particionar.py` distintos, `main/tensorflow_implementation` vacía.
- El script que generó los resultados es `finetune/messi.py`; su lanzador `messi.sh` llama a `finetune_yolo_obb.py`, que no existe.
- El README describe YOLOv8 y una GPU de 24 GB, no lo que se hizo.
- Git versiona `.pt`, `.onnx` y `__pycache__`; `.git` pesa ~209 MB. Restos de compilación LaTeX en `paper/`.
- `data/dataset_2500/images` tiene 2515 jpg para 2500 filas en el CSV (15 imágenes sobrantes por aclarar).

## Etapa 0: línea base reproducible
- [x] Rama creada, versiones del entorno registradas (Ultralytics 8.4.211, torch 2.15, ncnn 1.0.20261020).
- [x] Re-evaluar los checkpoints existentes y confirmar las Tablas 3-4 (dif. máx. 0.0008). Tabla 5 (AP por clase) pendiente de verificar.
- [x] Script del benchmark de la Pi: perdido; reescrito como `scripts/bench_pi.py`. **Pendiente: correrlo en la Pi.**
- Hecho cuando: un comando reproduce las tablas del paper (`scripts/repro_baseline.py`).

## Etapa 1: ordenar el repo  ✅ (commit `e605d7f`)
- Estructura: `src/swole_yolo/` (convert, split, train, evaluate, bench_pi), `configs/`, `scripts/`, `results/`, `paper/`, `docs/`, `archive/`.
- Un solo proyecto `uv` con `pyproject.toml`; fusionar los `particionar.py`; renombrar `messi.py`.
- Sacar pesos y `__pycache__` de git (Releases o DVC). Reescribir el historial queda como decisión aparte (afecta a los 4 coautores).
- README nuevo con el flujo completo.
- Hecho cuando: la Etapa 0 da los mismos números con el código nuevo.

## Etapa 2: guía principiante a experto + docs en Zensical  ✅
Sitio en `docs/` (config `zensical.toml`); 8 capítulos + glosario + 3 páginas del proyecto.
Imágenes y redes → detección (IoU, NMS, anchor-free) → métricas (P, R, AP, mAP, matriz de confusión) → OBB (parametrización, ecuación 1, DOTA) → YOLO11 y transfer learning → datos (fuga entre frames, split por clip, desbalance) → edge (ONNX, NCNN, latencia vs throughput, P95, throttling, cuantización) → diseño de experimentos y lectura de papers. Cada tema enlaza al código donde aparece.

## Etapa 3: afinar el paper actual  ✅ (salvo la decisión pendiente)
Hecho en `paper/simbig_paper.tex` (compila a 12 páginas):
- [x] Figs. 2 y 3 regeneradas sobre **test** (`scripts/make_paper_figures.py`); la Tabla 5 (AP por clase) verificada contra el checkpoint.
- [x] Punto de vista corregido: vistas aéreas cenitales (no "oblicuas" ni "cámaras elevadas").
- [x] Revisor 2: keywords dentro del abstract, párrafo de organización, qué es NCNN y por qué se eligió, pipeline de conversión, descripción de la arquitectura, URL del código.
- [x] Título y afirmaciones moderados ("data size", sin "controlled"); un solo run y test distinto por subset declarados en abstract, resultados y conclusión.
- [x] Tabla 3: eliminadas las columnas de FPS de escritorio (ruido); añadido el tamaño de test por fila.
- [x] Declarados: subconjuntos anidados y 1 088 clips, 81 % de cajas casi alineadas a ejes, objetos pequeños, convención de etiquetado, mejor checkpoint en épocas 43–50, ruido val vs test, versiones/logs no retenidos.
- **Decisión pendiente (del equipo): subconjunto de 3 000.** Opciones: (a) incluirlo en Tabla 1/3/Fig. 1 con su resultado (0.885 mAP@50, test de 452 imágenes) y explicar la no monotonía; (b) mantener la exclusión pero **declararla** en el texto con una razón verificable. Recomendado (a): la razón documentada para excluirlo no se sostiene con los datos.
- Pendiente antes de enviar: volver a medir la Pi con `scripts/bench_pi.py` (Tabla 6 proviene de un script perdido); comprobar que el repositorio sea público y contenga el estado final; ajustar la longitud al formato de la sede.
- Opcional: preprint en arXiv.

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
