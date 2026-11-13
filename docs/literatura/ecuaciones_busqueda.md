# Ecuaciones de búsqueda para la revisión manual

Guía para repetir (y completar) la revisión de literatura de forma **sistemática y registrable**, en Scopus, Web of Science, IEEE Xplore, Google Scholar y arXiv. La búsqueda hecha por el agente ([revisión](revision.md)) fue de unas 85 consultas sin protocolo: las afirmaciones "no encontré X" deben comprobarse con estas ecuaciones antes de escribir "no existe" en un paper.

## 1. Cómo usar esta página

1. Elige el **tema** (§3) y copia su ecuación en **Scopus** (referencia principal; la sintaxis se traduce a las demás bases en §4).
2. Aplica los **filtros**: 2021–2026 (para clásicos, quita el filtro), artículos de revista y de congreso, inglés y español.
3. **Registra** cada búsqueda (§6): fecha, base, ecuación exacta, número de resultados.
4. Cribado en dos pasos (título y abstract → texto completo) con los criterios de §5.
5. **Bola de nieve** (§7) desde las fuentes clave.

!!! tip "Si una ecuación devuelve demasiado o muy poco"
    - **> 1 000 resultados:** añade un bloque con `AND` (p. ej., `Aerial` o `Edge`) o restringe a `TITLE(...)`.
    - **< 20 resultados:** quita un bloque o agrega sinónimos con `OR`. En temas de nicho (OBB + Raspberry Pi) pocos resultados **son el hallazgo**: documenta la cifra.

## 2. Bloques de conceptos (se combinan con `AND`)

Dentro de cada bloque los sinónimos van con `OR`. Se escriben en sintaxis de Scopus (`*` = comodín; las frases entre comillas dobles).

| Bloque | Ecuación |
| --- | --- |
| **OBB** | `("oriented object detection" OR "rotated object detection" OR "oriented bounding box*" OR "rotated bounding box*" OR "arbitrary-oriented" OR "oriented detector*" OR "YOLO-OBB" OR "OBB detection")` |
| **HBB** | `("horizontal bounding box*" OR "axis-aligned" OR "HBB")` |
| **YOLO** | `(YOLO OR YOLOv8 OR YOLOv10 OR YOLO11 OR YOLOv11 OR YOLO12 OR YOLOv12 OR YOLO26 OR "You Only Look Once")` |
| **Detección** | `("object detection" OR "vehicle detection" OR detector* OR "deep learning")` |
| **Vehículo** | `(vehicle* OR car OR cars OR traffic OR motorcycle* OR motorbike* OR "motor-cycle*" OR tricycle* OR mototaxi* OR "moto-taxi*" OR "auto-rickshaw*" OR "tuk-tuk*" OR minibus* OR microbus* OR bus OR truck*)` |
| **Aéreo** | `(UAV OR UAVs OR "unmanned aerial vehicle*" OR drone* OR aerial OR overhead OR nadir OR "top-down" OR "bird's-eye" OR "bird's eye" OR "remote sensing")` |
| **Pequeños** | `("small object*" OR "tiny object*" OR "small target*" OR "small-object")` |
| **Borde** | `("edge computing" OR "edge AI" OR "edge device*" OR "edge deployment" OR embedded OR "Raspberry Pi" OR Jetson OR NCNN OR TFLite OR "TensorFlow Lite" OR "ONNX Runtime" OR TensorRT OR OpenVINO OR "on-device" OR "resource-constrained")` |
| **Rendimiento** | `(latency OR "inference time" OR FPS OR "frames per second" OR throughput OR benchmark* OR "real-time" OR "thermal throttling" OR quantiz*)` |
| **Datos** | `("data scaling" OR "scaling law*" OR "learning curve*" OR "data efficiency" OR "sample size" OR "training set size" OR "limited data" OR "small dataset*" OR "few-shot" OR "fine-tuning" OR "transfer learning" OR pretrain*)` |
| **Fuga** | `("data leakage" OR "information leakage" OR "train-test leakage" OR "train-test contamination" OR "data contamination")` |
| **Video** | `(video* OR frame* OR "temporal correlation" OR "spatiotemporal")` |
| **Seguimiento** | `("multi-object tracking" OR "multiple object tracking" OR MOT OR ByteTrack OR "BoT-SORT" OR DeepSORT OR "OC-SORT" OR "vehicle tracking")` |
| **Conteo** | `("vehicle counting" OR "traffic counting" OR "traffic flow" OR "traffic monitoring" OR "traffic analysis")` |
| **Métricas** | `("average precision" OR mAP OR "intersection over union" OR IoU OR ProbIoU OR "probabilistic IoU" OR "Gaussian bounding box*" OR "evaluation metric*")` |
| **Región** | `("Latin America*" OR Peru OR Lima OR Colombia OR Medell?n OR Mexico OR Brazil OR Chile OR Argentina OR "developing countr*" OR "Global South")` |
| **Varianza** | `(variance OR "random seed*" OR reproducib* OR "statistical significance" OR "confidence interval*" OR "multiple runs")` |

## 3. Ecuaciones por tema (Scopus)

Cada ecuación se envuelve así: `TITLE-ABS-KEY( … ) AND PUBYEAR > 2020 AND ( LIMIT-TO ( DOCTYPE , "ar" ) OR LIMIT-TO ( DOCTYPE , "cp" ) )`. Para brevedad se muestra solo el cuerpo con los bloques de §2 (reemplázalos por su contenido).

### A. Detección orientada y YOLO-OBB

| ID | Cuerpo | Qué busca |
| --- | --- | --- |
| A1 | `OBB AND (YOLO OR "real-time" OR lightweight OR "one-stage")` | YOLO y detectores livianos orientados |
| A2 | `OBB AND ("loss function*" OR loss) AND (Gaussian OR Wasserstein OR "Kullback-Leibler" OR ProbIoU OR "angle periodicity" OR "boundary discontinuity" OR "angular periodicity")` | Pérdidas para ángulo (GWD, KLD, ProbIoU, CSL) |
| A3 | `OBB AND DOTA AND (benchmark OR "state-of-the-art" OR comparison)` | Líneas base en DOTA |
| A4 | `OBB AND vehicle* AND Aéreo` | OBB aplicado a vehículos aéreos |

### G. **OBB vs. caja horizontal** (novedad C1: verificar si ya existe)

| ID | Cuerpo | Nota |
| --- | --- | --- |
| G1 | `OBB AND HBB AND (comparison OR versus OR vs OR ablation OR "compared with" OR "compared to")` | Comparación directa |
| G2 | `OBB AND HBB AND vehicle* AND Aéreo AND (accuracy OR mAP OR "average precision")` | Con vehículos aéreos |
| G3 | `("oriented bounding box*" OR "rotated bounding box*") AND ("axis-aligned" OR "horizontal") AND (benefit* OR advantage* OR "does not" OR "no improvement" OR "limited")` | Busca estudios que cuestionan el beneficio |

!!! warning "Sobre la novedad de C1"
    La [hoja de ruta](../mejora/plan_de_mejora.md) apoya C1 en que **no se encontró** un estudio controlado OBB vs. HBB con objetos casi alineados. Antes de sostenerlo en un paper, G1–G3 deben revisarse a mano en Scopus **y** Web of Science (≥ 50 primeros resultados de cada una, por relevancia), y repetirse en Google Scholar.

### B. Vehículos desde dron, objetos pequeños y regionales

| ID | Cuerpo |
| --- | --- |
| B1 | `Vehículo AND Aéreo AND Detección AND (benchmark OR dataset)` |
| B2 | `Vehículo AND Aéreo AND Pequeños AND (YOLO OR detector*)` |
| B3 | `Vehículo AND Aéreo AND Región` |
| B4 | `(mototaxi* OR "moto-taxi*" OR "auto-rickshaw*" OR "tuk-tuk*" OR "three-wheeler*") AND (detection OR classification) AND (aerial OR drone OR UAV OR "traffic")` |
| B5 | `(minibus* OR microbus* OR "paratransit" OR "informal transport") AND ("object detection" OR "deep learning")` |
| B6 | `"MTC SMART Challenge" OR ("SMART Challenge" AND Peru)` *(literatura gris: probar también en Google Scholar y en la web)* |

### C. Despliegue en el borde

| ID | Cuerpo | Qué busca |
| --- | --- | --- |
| C1 | `YOLO AND Borde AND Rendimiento` | Benchmarks de YOLO en borde |
| C2 | `"Raspberry Pi" AND (YOLO OR "object detection") AND (latency OR FPS OR "inference time")` | Pi específica |
| C3 | `("thermal throttling" OR "thermal behavio*r" OR "CPU temperature" OR "sustained performance") AND ("Raspberry Pi" OR "edge device*" OR embedded) AND (inference OR "neural network*")` | Protocolo térmico |
| C4 | `quantiz* AND YOLO AND ("INT8" OR "8-bit" OR "post-training") AND Borde` | Cuantización |
| C5 | `(NCNN OR "ONNX Runtime" OR TFLite OR OpenVINO) AND (comparison OR benchmark*) AND ("Raspberry Pi" OR ARM OR "edge device*")` | Comparación de runtimes |
| **C6** | `OBB AND Borde` y `OBB AND ("Raspberry Pi" OR ARM OR "edge device*" OR embedded OR Jetson)` | **OBB en borde (novedad C4: verificar que no existe)** |

### D. Datos: escalado, preentrenamiento, fuga y varianza

| ID | Cuerpo |
| --- | --- |
| D1 | `Datos AND Detección AND ("learning curve*" OR "scaling law*" OR "sample size" OR "training set size")` |
| D2 | `(pretrain* OR "transfer learning") AND ("object detection") AND (aerial OR "remote sensing") AND (scratch OR "ImageNet" OR "COCO" OR "DOTA")` |
| D3 | `Fuga AND Video AND (detection OR "object detection" OR "deep learning")` |
| D4 | `("data leakage" OR "information leakage" OR split*) AND (video* OR frame*) AND ("object detection" OR "deep learning") AND (cluster* OR group* OR "scene-level" OR "video-level")` |
| D5 | `Varianza AND ("deep learning" OR "object detection") AND (benchmark* OR comparison)` |

### E. Conteo y seguimiento

| ID | Cuerpo |
| --- | --- |
| E1 | `Seguimiento AND Aéreo AND Vehículo` |
| E2 | `Conteo AND Aéreo AND (YOLO OR detector*)` |
| E3 | `Conteo AND Seguimiento AND ("low frame rate" OR "frame rate" OR "frame skipping" OR "temporal sampling")` |
| E4 | `Seguimiento AND OBB` |

### F. Metodología de evaluación

| ID | Cuerpo |
| --- | --- |
| F1 | `Métricas AND OBB AND ("polygon IoU" OR "rotated IoU" OR "skew IoU" OR "evaluation protocol")` |
| F2 | `("average precision" OR mAP) AND ("object detection") AND (interpolation OR "evaluation protocol" OR "metric comparison")` |
| F3 | `"error analysis" AND "object detection" AND (TIDE OR "error diagnosis" OR "false positive*")` |

## 4. Traducción a otras bases

| Elemento | Scopus | Web of Science | IEEE Xplore | Google Scholar | arXiv |
| --- | --- | --- | --- | --- | --- |
| Campos | `TITLE-ABS-KEY( … )` | `TS=( … )` (título, abstract, keywords) | `"All Metadata":…` o `"Abstract":…` | texto libre; `allintitle:` para título | campo `abs` o `ti` en búsqueda avanzada |
| Años | `PUBYEAR > 2020` | `PY=(2021-2026)` | filtro "Year" | "Personalizar intervalo" | filtro de fecha |
| Frase | `"…"` | `"…"` | `"…"` | `"…"` | `"…"` |
| Comodín | `*` | `*` | `*` (limitado) | no soporta | no soporta |
| Proximidad | `W/5` | `NEAR/5` | `NEAR/5` | — | — |
| Tipo | `DOCTYPE` | `DT=(Article OR Proceedings Paper)` | "Content type" | — | — |

!!! info "Límites prácticos"
    - **IEEE Xplore** rechaza consultas muy largas o con demasiados operadores anidados: divide un bloque en varias búsquedas y une los resultados en Zotero.
    - **Google Scholar** no admite comodines ni ecuaciones largas: usa 2–3 frases clave. Revisa solo los primeros ~100 resultados por relevancia y las citas "Citado por".
    - **arXiv** no está indexado igual: sirve para preprints recientes (YOLO26, RT-DETR orientado), que Scopus aún no tiene.

### Ejemplos listos para pegar

**Scopus · A1 (YOLO orientado)**

```text
TITLE-ABS-KEY ( ( "oriented object detection" OR "rotated object detection" OR "oriented bounding box*" OR "rotated bounding box*" OR "arbitrary-oriented" OR "YOLO-OBB" ) AND ( YOLO OR YOLOv8 OR YOLO11 OR YOLOv11 OR YOLO12 OR YOLOv12 OR YOLO26 OR "real-time" OR lightweight ) ) AND PUBYEAR > 2020 AND ( LIMIT-TO ( DOCTYPE , "ar" ) OR LIMIT-TO ( DOCTYPE , "cp" ) )
```

**Web of Science · G1 (OBB vs. HBB)**

```text
TS=(("oriented bounding box*" OR "rotated bounding box*" OR "oriented object detection" OR "rotated object detection") AND ("horizontal bounding box*" OR "axis-aligned" OR "HBB") AND (comparison OR versus OR ablation OR "compared with")) AND PY=(2021-2026) AND DT=(Article OR Proceedings Paper)
```

**Scopus · C6 (OBB en borde)**

```text
TITLE-ABS-KEY ( ( "oriented object detection" OR "rotated object detection" OR "oriented bounding box*" OR "rotated bounding box*" OR "YOLO-OBB" ) AND ( "Raspberry Pi" OR Jetson OR "edge device*" OR "edge computing" OR embedded OR NCNN OR "on-device" ) )
```

**IEEE Xplore · C3 (térmica en Raspberry Pi)**

```text
("Raspberry Pi") AND ("thermal throttling" OR "CPU temperature" OR "sustained performance") AND ("inference" OR "neural network")
```

**Google Scholar · fuga en video**

```text
"data leakage" "object detection" video frames split "spatiotemporal" OR "video-level"
```

**Scopus · B4 (vehículos regionales)**

```text
TITLE-ABS-KEY ( ( mototaxi* OR "moto-taxi*" OR "auto-rickshaw*" OR "tuk-tuk*" OR "three-wheeler*" ) AND ( detection OR classification ) AND ( aerial OR drone OR UAV OR traffic ) )
```

## 5. Criterios de inclusión y exclusión

| Incluir | Excluir |
| --- | --- |
| Detección de vehículos con cajas (horizontales u orientadas), o despliegue/borde de detectores, o metodología de evaluación/datos aplicable | Solo clasificación o segmentación sin relación con el tema |
| 2021–2026 (clásicos fundacionales sin filtro de año) | Duplicados y versiones preliminares ya cubiertas por la versión de revista |
| Revistas y congresos arbitrados; preprints de arXiv **marcados como tales** | Resúmenes de una página, patentes, notas de prensa (salvo para el dataset del MTC, como literatura gris) |
| Texto completo accesible o abstract completo | Sin posibilidad de verificar autores, año y venue |
| Inglés o español | — |

## 6. Registro de búsquedas (plantilla)

Copia esta tabla a una hoja de cálculo o a un archivo del repositorio y complétala **por cada búsqueda**:

| Fecha | Base | ID | Ecuación exacta (copiada) | Filtros | Resultados | Revisados | Incluidos | Notas |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| *AAAA-MM-DD* | Scopus | A1 | *…* | 2021+, ar+cp | | | | |

Flujo de cribado (estilo PRISMA simplificado): **identificados → sin duplicados → cribados por título/abstract → texto completo evaluado → incluidos**. Anota el número en cada paso y el motivo de cada exclusión en texto completo.

## 7. Bola de nieve (cuando la búsqueda no basta)

Desde estas fuentes clave, revisar **hacia atrás** (sus referencias) y **hacia adelante** (quién las cita, con "Citado por" en Scopus o Google Scholar):

| Fuente semilla | Para encontrar |
| --- | --- |
| Lepin 2026 (UTUAV, *Drones*) | Datasets y líneas base de dron en Latinoamérica; fuga de datos |
| Figueiredo y Mendes 2024 (*IEEE Access*) | Más trabajo sobre fuga en video |
| Benoit-Cattin 2020 (*Electronics*) | Benchmarks térmicos en Raspberry Pi |
| Rey 2025 (*Electronics*) | Detección en borde para dron con cuantización |
| Llerena 2021 (ProbIoU) | Evaluación de cajas orientadas con gaussianas |
| Ye y Kyrkou 2026 (*IEEE T-ITS*) | Revisión de detección vehicular desde UAV |
| Ramos-Sanchez 2026 (*J. Imaging*) | Trabajo peruano y regional |

## 8. Alertas para mantenerse al día

Configura alertas con las ecuaciones **A1, C6, G1 y B3** (Scopus, Google Scholar y arXiv): el área cambia cada pocos meses (YOLOv12 en 2025, YOLO26 en 2026).

!!! warning "Antes de escribir una afirmación de ausencia"
    Para "no existe X" en un paper: ejecuta la ecuación correspondiente en **al menos dos bases** (Scopus y Web of Science), anota fecha y número de resultados, y revisa a mano los primeros resultados por relevancia. La redacción prudente es *"hasta donde sabemos, y según una búsqueda en [bases] realizada en [fecha]"*.
