# Revisión de literatura (2021-2026) para la reformulación del paper YOLO11n-OBB / MTC SMART Challenge

Fecha de la búsqueda: 2026-11-11. Método: búsquedas web y lectura de páginas de arXiv, CVF, PMLR, Crossref y editoriales. Las claves BibTeX apuntan a `paper/references_new.bib`.

!!! success "Verificación independiente (hecha por el equipo tras la búsqueda)"
    Esta revisión la elaboró un agente de IA (Fable 5.2, esfuerzo medio). Antes de usarla se comprobó cada entrada del `.bib` contra Crossref (DOI) y la API de arXiv (identificador):

    - **66 de 66 fuentes existen** (63 resueltas automáticamente + 3 comprobadas aparte: TFA y Pineau por su URL oficial; Gellida-Coutino por IEEE Xplore/DOAJ).
    - **57 coincidían exactamente** en título y año. Se corrigieron: el título y los autores de `alqahtani2024edgebench` (estaban parafraseados), el año faltante de `zheng2024citysim` y la nota de `gellidacoutino2025occluded`. Cuatro diferencias de año son normales (arXiv 2022 vs. congreso 2023, publicación en línea 2025 vs. número 2026).
    - Se leyeron los abstracts de las fuentes que cita el paper (UTUAV, Ramos-Sanchez, Benoit-Cattin, Rey 2025, Alqahtani, YOLO26, Lin y Su, Wang 2025, RTMDet, ProbIoU, Bouthillier): **dicen lo que esta revisión afirma**.
    - Se confirmó en el código de Ultralytics 8.4.211 que la validación OBB usa ProbIoU (`batch_probiou` en `obb/val.py`).
    - **No** se verificó que "no existe X" (las afirmaciones de ausencia son de ~85 consultas, no de una búsqueda sistemática) ni los cuartiles de las revistas.

## Cómo leer este documento

- **verificado: sí** = título, autores, año, venue y DOI/URL confirmados en una búsqueda o fetch, y leí al menos el resumen (abstract) o un registro bibliográfico oficial (arXiv, Crossref, CVF, PMLR, JMLR).
- **verificado: parcial** = la existencia y el venue están confirmados, pero falta algún dato (DOI, volumen, lista completa de autores) o el contenido solo lo vi en un listado/snippet de terceros. Se indica qué falta.
- Ninguna fuente fue leída a texto completo: **todo lo que se afirma sobre contenido proviene del abstract o de resúmenes de búsqueda**. Antes de citar una cifra concreta en el paper hay que abrir el PDF.
- Las cifras (mAP, FPS) llevan entre paréntesis de dónde salieron. No hay cifras de comparación inventadas; donde no encontré un número, lo digo.
- Las referencias que el paper ya cita (DOTA, YOLO, COCO, UA-DETRAC, BDD100K, VisDrone 2018, RoI Transformer, R3Det, Oriented R-CNN, NCNN, docs de Ultralytics, Santos 2024) no se repiten como nuevas.
- Fuentes descartadas o dudosas: ver `docs/literatura/no_verificado.md`.

Recuento de fuentes nuevas: A = 21, B = 17, C = 8, D = 8, E = 9, F = 3 (más 4 reutilizadas de otros temas). Total = 66 (62 "sí", varias con una salvedad anotada entre paréntesis, y 4 "parcial").

---

## A. Detección de objetos orientados y YOLO-OBB

### Síntesis

**Qué se sabe.** La detección orientada moderna se organiza en tres líneas. (1) *Arquitecturas específicas para imágenes aéreas*: alineación de características (S2A-Net), equivarianza a la rotación (ReDet) y campos receptivos grandes y selectivos (LSKNet, 81.85 % mAP en DOTA-v1.0 según su abstract). (2) *Detectores de una etapa en tiempo real con cabeza rotada*: RTMDet-R (81.33 % AP en DOTA-v1.0 con preentrenamiento COCO, abstract), PP-YOLOE-R (78.14 mAP con una sola escala; 69.8 FPS en la variante s sobre RTX 2080 Ti con TensorRT FP16, abstract) y la familia Ultralytics (YOLOv8/YOLO11/YOLO12/YOLO26 con tarea OBB). (3) *Transformers en tiempo real*: RT-DETR no modela la rotación; recién en 2026 aparece un RT-DETR orientado (O2-RTDETR-R50: 78.45 % AP50 en DOTA-v1.0 a 119 FPS en 2080 Ti, abstract).

El segundo eje es la **pérdida de regresión**. El IoU rotado exacto es poco amigable para el entrenamiento y la parametrización angular tiene discontinuidades de frontera (CSL convierte el ángulo en clasificación para evitarlas). La solución dominante es modelar cada caja como una gaussiana 2D y medir una distancia entre distribuciones: GWD (Wasserstein), KLD (Kullback-Leibler), KFIoU (producto de gaussianas) y ProbIoU (distancia de Hellinger). **ProbIoU es la que usa Ultralytics** tanto en la pérdida como en el emparejamiento de la validación OBB, lo que tiene consecuencias metodológicas (tema F).

Sobre la **familia Ultralytics** hay un problema bibliográfico: YOLOv8 y YOLO11 no tienen artículo original revisado por pares; lo citable son revisiones de terceros (Terven 2023; Khanam y Hussain 2024; Hidayatullah 2025) y, desde junio de 2026, el artículo propio de Ultralytics sobre YOLO26, que además cambia la definición del ángulo y añade una pérdida angular sensible a la relación de aspecto. YOLOv10 y YOLOv12 sí tienen preprint de sus autores. Para un paper enviado a fines de 2026, **YOLO11n ya no es el estado del arte de su propia familia**: un revisor pedirá, como mínimo, YOLO12n-OBB y YOLO26n-OBB como comparación.

**Qué falta.** No encontré ningún estudio controlado que compare OBB contra cajas horizontales (HBB) con el mismo backbone, los mismos datos y las mismas métricas **cuando la mayoría de los objetos está casi alineada con los ejes**. La literatura justifica OBB con argumentos geométricos (menos fondo, menos solapamiento en escenas densas, rumbo del vehículo) y con datasets de orientación arbitraria; los trabajos de vehículos desde dron que usan OBB (Lin y Su 2022; Wang 2025; DRIFT) lo motivan cualitativamente. Tampoco encontré reportes de detectores OBB ejecutados en CPU ARM (Raspberry Pi): los benchmarks de velocidad de la línea OBB son todos en GPU (2080 Ti, T4).

### Tabla A

| Ref. corta | Venue | Método | Dataset | Métricas / hardware (fuente) | Relevancia y diferencia con nuestro trabajo | Clave BibTeX | Verificado |
|---|---|---|---|---|---|---|---|
| Llerena 2021 | arXiv 2106.06072 (ref. de revista en arXiv: IEEE TIP 33, 2024) | Cajas gaussianas y ProbIoU (distancia de Hellinger) | COCO y otros públicos | Sin cifras en el abstract | Es la pérdida y la métrica de emparejamiento de YOLO-OBB en Ultralytics; hay que citarla y discutir que es una aproximación | `llerena2021probiou` | sí |
| Yang 2021a | ICML 2021 | GWD: Wasserstein entre gaussianas | DOTA y otros | Sin cifras en lo leído | Antecedente directo de las pérdidas gaussianas; señala utilidad con cajas que no se solapan (objetos pequeños) | `yang2021gwd` | sí |
| Yang 2021b | NeurIPS 2021 | KLD entre gaussianas | DOTA y otros | Sin cifras en lo leído | Alternativa a ProbIoU; candidata para una ablación de pérdida | `yang2021kld` | sí |
| Yang 2023 | ICLR 2023 | KFIoU (aprox. de SkewIoU por producto gaussiano) | 2D y 3D | Sin cifras en lo leído | Tercera pérdida gaussiana; solo se confirmó el primer autor | `yang2023kfiou` | parcial (lista de autores) |
| Yang y Yan 2020 | ECCV 2020 | CSL: ángulo como clasificación circular | DOTA, HRSC2016, ICDAR2015, MLT | Sin cifras en lo leído | Explica la discontinuidad angular; útil para justificar por qué el 81 % casi alineado no es trivial (frontera en 0°/90°) | `yang2020csl` | sí |
| Han 2021 | CVPR 2021 | ReDet: equivarianza a la rotación + RiRoI Align | DOTA-v1.0/1.5, HRSC2016 | +1.2/+3.5/+2.6 mAP con 60 % menos parámetros (repositorio oficial) | Línea de base de dos etapas; no está pensada para borde | `han2021redet` | sí |
| Han 2022 | IEEE TGRS 60 | S2A-Net: alineación de características | DOTA, HRSC2016 | Sin cifras en lo leído | Línea de base de una etapa clásica | `han2022s2anet` | sí |
| Li 2023 | ICCV 2023 | LSKNet: kernels grandes selectivos | DOTA-v1.0, HRSC2016, FAIR1M | 81.85 % mAP DOTA-v1.0 (abstract) | Techo de precisión reciente en DOTA; backbone más pesado que un modelo "n" | `li2023lsknet` | sí |
| Lyu 2022 | arXiv 2212.07784 | RTMDet / RTMDet-R | COCO, DOTA | 81.33 % AP DOTA-v1.0 (abstract) | **Comparador más natural fuera de Ultralytics**: una etapa, en tiempo real, con variantes tiny | `lyu2022rtmdet` | parcial (lista de autores) |
| Wang 2022 | arXiv 2211.02386 | PP-YOLOE-R, sin anclas | DOTA-v1.0 | 78.14 mAP (l, una escala); 69.8 FPS (s, 2080 Ti, TensorRT FP16) (abstract) | Segundo comparador en tiempo real; ecosistema Paddle | `wang2022ppyoloer` | sí |
| Zhou 2022 | ACM MM 2022 | MMRotate (toolbox y benchmark) | DOTA y otros | No aplica | Herramienta para entrenar comparadores (RTMDet-R, S2A-Net, Oriented R-CNN) con protocolo común | `zhou2022mmrotate` | sí |
| Khanam y Hussain 2024 | arXiv 2410.17725 | Descripción de YOLO11 (C3k2, SPPF, C2PSA) | No aplica | No aplica | Referencia citable para la arquitectura YOLO11; no revisada por pares | `khanam2024yolov11` | sí |
| Tian 2025 | arXiv 2502.12524 | YOLOv12, centrado en atención | COCO | YOLOv12-N 40.6 % mAP, 1.64 ms en T4 (abstract) | Sucesor directo; comparador obligatorio | `tian2025yolov12` | sí |
| Wang 2024 | arXiv 2405.14458 | YOLOv10, sin NMS | COCO | 1.8x más rápido que RT-DETR-R18 con AP similar (abstract) | Antecedente del diseño sin NMS; sede NeurIPS no confirmada | `wang2024yolov10` | sí (como arXiv) |
| Jocher 2026 | arXiv 2606.03748 | YOLO26: sin NMS, sin DFL, OBB incluido | COCO | 40.9-57.5 mAP, 1.7-11.8 ms T4 TensorRT (abstract). Mejora OBB en DOTA y pérdida angular: solo en resúmenes de terceros | **Estado del arte de la propia familia en 2026**; primer artículo de los autores de Ultralytics | `jocher2026yolo26` | sí (cifras OBB: parcial) |
| Zhao 2024 | CVPR 2024 | RT-DETR | COCO | R50: 53.1 % AP, 108 FPS en T4 (abstract) | Comparador transformer (solo HBB) | `zhao2024rtdetr` | sí |
| Ding 2026 | IEEE TGRS 2026 | O2-RTDETR / O2-DFINE / O2-DEIM | DOTA-v1.0 | 78.45 % AP50 a 119 FPS (2080 Ti) (abstract) | Muestra que el DETR orientado en tiempo real es de 2026; no hay evidencia en CPU | `ding2026o2rtdetr` | sí |
| Terven 2023 | Mach. Learn. Knowl. Extr. 5(4) | Revisión YOLOv1-YOLOv8 | No aplica | No aplica | Revisión latinoamericana (México) citable para YOLOv8 | `terven2023yoloreview` | sí |
| Hidayatullah 2025 | arXiv 2501.13400 (versión de revista en Jurnal RESTI 2026, con otro título) | Comparación arquitectónica YOLOv8-YOLO11 | No aplica | No aplica | Complementa a Khanam para describir bloques | `hidayatullah2025yolov8to11` | sí |
| Lin y Su 2022 | Sensors 22(21) | Detector orientado sobre YOLOv4 para aforos | Aéreo, vehículos | ">25 % más rápido" que otros enfoques (abstract) | Antecedente directo: YOLO orientado para vehículos en estudios de tráfico; no borde, no análisis de datos | `lin2022orientedyolov4` | sí |
| Wang 2025 | Remote Sensing 17(3) | Red de kernel selectivo asimétrico + dataset FRVehicle con OBB | Dron, rampas de autopista | Sin cifras en lo leído | OBB de vehículos desde dron para trayectorias; un solo tipo de vía, sin vehículos regionales | `wang2025frvehicle` | sí |

---

## B. Vehículos en imágenes de dron/aéreas, objetos pequeños y tráfico regional

### Síntesis

**Qué se sabe.** Los benchmarks canónicos de dron son VisDrone (resumen de los desafíos 2018-2020 en TPAMI 2022), UAVDT (unos 80 000 frames de 10 horas de video, con atributos) y DroneVehicle (28 439 pares RGB-infrarrojo con cajas orientadas). Para objetos diminutos están AI-TOD (700 621 instancias, tamaño medio de 12.8 px, según el abstract) y SODA-A/SODA-D. Las estrategias que mejoran objetos pequeños son conocidas: más resolución, cabezas adicionales de alta resolución (TPH-YOLOv5, Drone-YOLO), inferencia por recortes (SAHI: +5.1 a +6.8 AP solo con inferencia por recortes en VisDrone/xView, abstract) y métricas de asignación tolerantes a pequeños desplazamientos (NWD). Hay además una línea aplicada de transporte que extrae **trayectorias** desde dron con cajas rotadas: CitySim (1 140 minutos, 12 sitios) y DRIFT (Daejeon, unos 250 m de altura, 81 699 trayectorias; los resúmenes indican YOLO11 con OBB y ByteTrack). La revisión de Ye y Kyrkou (T-ITS 2026) confirma que el borde y el tiempo real son retos abiertos de esta línea.

**Latinoamérica.** La evidencia regional existe pero es escasa y muy reciente:

- **UTUAV** (Drones 2026): dataset de dron en Medellín, a 100 y 120 m, tres clases (auto, moto, vehículo grande), con líneas base YOLOv8/YOLO11/YOLO12/RT-DETR. Su abstract discute explícitamente la **fuga de datos** cuando frames de un mismo video se reparten entre particiones. Es el trabajo más cercano al nuestro y hay que citarlo y diferenciarse.
- **YucaMex-Drone** (IJITSR 2026): 4 448 imágenes de dron de ciudades de Yucatán, ocho categorías, detección más seguimiento para conteo.
- **Lima** (Journal of Imaging 2026): detectores SSD reentrenados con tráfico de Lima sobre Jetson Orin NX, pero con cámara a nivel de calle, no aérea.

**Qué falta.** No encontré ningún dataset o artículo indexado de vista aérea con clases propias del transporte peruano o andino (**mototaxi, microbús/combi, minibús, bus articulado**) ni con OBB en Latinoamérica. UTUAV usa tres clases gruesas y cajas horizontales (por confirmar en el texto completo). El dataset del MTC SMART Challenge solo aparece en notas de prensa: no tiene todavía una publicación descriptiva citable, lo que es una oportunidad (documentarlo) y un riesgo (licencia y disponibilidad).

### Tabla B

| Ref. corta | Venue | Método | Dataset | Métricas / hardware (fuente) | Relevancia y diferencia con nuestro trabajo | Clave BibTeX | Verificado |
|---|---|---|---|---|---|---|---|
| Sun 2022 | IEEE TCSVT 32(10) | Fusión RGB-IR con incertidumbre | DroneVehicle (OBB) | 28 439 pares, 953 087 instancias (arXiv) | Benchmark OBB de vehículos desde dron; candidato a preentrenamiento o evaluación cruzada | `sun2022dronevehicle` | sí |
| Du 2018 | ECCV 2018 | Benchmark detección y seguimiento | UAVDT | ~80 000 frames, 10 h (abstract) | Clásico; solo vehículos, HBB | `du2018uavdt` | sí (páginas sin confirmar) |
| Wang 2021a | ICPR 2020 (publ. 2021) | M-CenterNet + dataset | AI-TOD | Tamaño medio 12.8 px (abstract) | Nuestra moto (~13 px de lado corto) cae en el régimen "tiny" | `wang2021aitod` | sí |
| Zhu 2022 | IEEE TPAMI 44(11) | Resumen de los desafíos VisDrone | VisDrone | No aplica | Referencia actualizada de VisDrone (el paper cita la de 2018) | `zhu2022visdronechallenge` | sí (lista de autores parcial) |
| Cheng 2023 | IEEE TPAMI 45(11) | Revisión y benchmarks de objetos pequeños | SODA-D, SODA-A | No aplica | Marco conceptual para definir "pequeño" y discutir resolución | `cheng2023soda` | sí |
| Akyon 2022 | ICIP 2022 | SAHI: inferencia y ajuste por recortes | VisDrone, xView | +6.8/+5.1/+5.3 AP solo inferencia (abstract) | Alternativa a subir `imgsz`; relevante para el compromiso precisión-latencia en la Pi | `akyon2022sahi` | sí |
| Wang 2021b | arXiv 2110.13389 | NWD para objetos diminutos | AI-TOD | +6.7 AP sobre la línea base (abstract) | Explica por qué IoU castiga a las motos; conecta con mAP@50-95 bajo en clases pequeñas | `wang2021nwd` | sí |
| Zhu 2021 | ICCVW 2021 | TPH-YOLOv5 | VisDrone | AP 39.43 % (abstract) | Receta estándar de YOLO para dron (cabeza extra) | `zhu2021tphyolov5` | sí |
| Zhang 2023 | Drones 7(8) | Drone-YOLO sobre YOLOv8 | VisDrone2019 | Variante de 5.25 M parámetros iguala a línea base de 9.66 M (resumen) | Ejemplo del tipo de paper que publica Drones; modifica arquitectura, nosotros no | `zhang2023droneyolo` | sí |
| Zhang 2025 | Sensors 25(11) | DSCW-YOLO sobre YOLOv11s | Dataset urbano propio | Mejor F1 entre comparados (registro DOAJ) | Ejemplo comparable en Sensors; HBB, sin borde | `zhang2025dscwyolo` | sí |
| **Lepin 2026** | Drones 10(1) | Dataset + líneas base YOLOv8/11/12, RT-DETR | **UTUAV (Medellín)** | Sin cifras en el abstract | **Competidor directo**: dron, Latinoamérica, motos, fuga de datos. Diferencias nuestras: OBB, 9 clases finas, borde | `lepin2026utuav` | sí |
| Lee 2025 | arXiv 2504.11019 | Dataset de trayectorias con OBB | DRIFT (Daejeon) | 81 699 trayectorias, ~250 m (abstract); YOLO11-OBB + ByteTrack (resumen de búsqueda) | Mismo detector base aplicado a trayectorias; sin borde ni estudio de datos | `lee2025drift` | sí (detector: parcial) |
| Zheng 2024 | Transportation Research Record | Dataset de trayectorias con cajas rotadas | CitySim | 1 140 min, 12 sitios (arXiv) | Justifica OBB por medidas de seguridad vial | `zheng2024citysim` | parcial (volumen y páginas) |
| Ramos-Sanchez 2026 | Journal of Imaging 12(1) | SSD-MobileNet / VGG16-SSD reentrenados | Tráfico de Lima (vista de calle) | Jetson Orin NX 16 GB (abstract) | **Antecedente peruano más cercano**; no aéreo, no OBB, GPU embebida | `ramossanchez2026jetsonlima` | sí |
| Salcedo-Delgado 2026 | Int. J. ITS Research | Detección + MOT para conteo | YucaMex-Drone (4 448 imágenes, 8 clases) | Sin cifras en lo leído | Conteo desde dron en ciudad latinoamericana; referencia para el tema E | `salcedodelgado2026yucatan` | sí |
| Ye y Kyrkou 2026 | IEEE T-ITS | Revisión con experimentos | Varios | No aplica | Revisión más reciente del área; respalda que borde y tiempo real son retos abiertos | `ye2026uavreview` | sí |
| Wang 2026 | arXiv 2604.01134 | Dataset de tráfico mixto | VRUD (Shenzhen) | 4 h 4K, 11 479 trayectorias de usuarios vulnerables (abstract) | Tráfico desordenado en "urban villages"; análogo asiático del problema | `wang2026vrud` | sí (detector: parcial) |

---

## C. Despliegue en el borde

### Síntesis

**Qué se sabe.** Hay una oferta amplia de benchmarks de YOLO en dispositivos de borde, pero casi todos reportan **una latencia media o FPS por configuración** y pocos caracterizan el estado térmico o la variabilidad:

- Alqahtani 2024 mide YOLOv8 n/s/m, EfficientDet-Lite y SSD en Raspberry Pi 3/4/5 (con y sin TPU) y Jetson Orin Nano, con energía, tiempo de inferencia y mAP.
- Rey 2025 (Electronics) compara YOLOv8n/s en Jetson Orin Nano/NX y Raspberry Pi 5 con cuantización posentrenamiento: 52 FPS en Orin NX y 65 FPS con INT8; la Pi 5 no alcanzó tiempo real (abstract).
- Santos 2024 (ya citado por el paper) reporta alrededor de 0.9 FPS con YOLOv7-tiny en Raspberry Pi 4B (registro de repositorio CAPES).
- Andika 2024 (JRTIP) propone un YOLOv7-tiny mejorado para vehículos en hardware de bajo costo.
- Aljami 2026 (Computers) estudia YOLOv8 en Jetson Orin NX: TensorRT un 17.7 % más rápido que PyTorch con lote 2 (abstract).

El trabajo metodológicamente más relevante es **Benoit-Cattin 2020**: 80 casos de larga duración en Raspberry Pi 4B (5 CNN, 4 frameworks, 2 sistemas operativos). El estrangulamiento térmico aparece en segundos con cargas pesadas; el enfriamiento activo mejora el rendimiento hasta cerca de 90 %, y sin ventilador la temperatura ambiente (22-36 °C) cambia el rendimiento hasta 27.7 % (abstract). Esto implica que un número como "300 ms a 544 px" **no es interpretable sin declarar refrigeración, temperatura, duración de la corrida y governor**.

Sobre cuantización, Jacob 2018 es la referencia fundacional de INT8; Karimov 2025 muestra en YOLO que INT8 estático da la mayor aceleración con pérdida moderada de mAP en datos limpios y mayor sensibilidad a degradaciones. MLPerf Inference fija el estándar de reporte por escenarios (single-stream, etc.) y percentiles de cola.

**Qué falta.** (1) No encontré mediciones publicadas de **modelos OBB** en Raspberry Pi ni en ningún CPU ARM. (2) No encontré un estudio de YOLO en Raspberry Pi que reporte conjuntamente P50/P95/P99, número de hilos, governor y temperatura; el vacío está en la intersección, no en cada pieza. (3) Las comparaciones de runtimes (NCNN, TFLite, ONNX Runtime, OpenVINO) sobre Pi 4 que encontré están en documentación de Ultralytics y blogs, no en artículos arbitrados. (4) No hallé trabajos latinoamericanos de borde con vista aérea.

### Tabla C

| Ref. corta | Venue | Método | Dataset | Métricas / hardware (fuente) | Relevancia y diferencia con nuestro trabajo | Clave BibTeX | Verificado |
|---|---|---|---|---|---|---|---|
| **Benoit-Cattin 2020** | Electronics 9(12) | 80 casos de inferencia prolongada | 5 CNN de clasificación | Raspberry Pi 4B; hasta ~90 % más rendimiento con enfriamiento activo; 27.7 % de variación por temperatura ambiente (abstract) | **Base metodológica del protocolo térmico**; no usa detectores YOLO ni NCNN (por confirmar) | `benoitcattin2020thermal` | sí |
| Rey 2025 | Electronics 14(3) | YOLOv8n/s con PTQ | Aplicación en dron | Orin NX 52 FPS, 65 FPS con INT8; Pi 5 sin tiempo real (abstract) | Comparable directo en hardware; HBB, sin Pi 4, sin protocolo térmico declarado en el abstract | `rey2025yoloedgedrone` | sí |
| Alqahtani 2024 | arXiv 2409.16808 | YOLOv8, EfficientDet-Lite, SSD | COCO | Pi 3/4/5 ± TPU, Jetson Orin Nano; energía, latencia, mAP (abstract) | Referencia multiplataforma; da el eje "otro hardware" que pidieron los revisores | `alqahtani2024edgebench` | sí |
| Andika 2024 | J. Real-Time Image Process. 21(3) | YOLOv7-tiny mejorado | Vehículos | mAP 57.94 % (snippet de búsqueda; no confirmado en el abstract) | Ejemplo comparable publicado en JRTIP | `andika2024yolov7tinyvehicle` | sí (cifra: parcial) |
| Aljami 2026 | Computers 15(2) | YOLOv8 en PyTorch vs TensorRT | Vehículos y paquetes | Jetson Orin NX; TensorRT +17.7 % con lote 2 (abstract) | Benchmark puro de variantes en una placa; valida que ese formato es publicable | `aljami2026yolov8orinnx` | sí |
| Karimov 2025 | arXiv 2508.19600 | FP32, FP16, UINT8 dinámico, INT8 estático | COCO | INT8 estático: mayor aceleración, pérdida moderada (abstract). Familia de modelo (YOLOv8 o YOLO12) difiere entre versiones | Sustento para un experimento de cuantización en NCNN | `karimov2025quantrobust` | sí |
| Jacob 2018 | CVPR 2018 | Cuantización entera y QAT | ImageNet, COCO | No aplica | Referencia fundacional de INT8 | `jacob2018quantization` | sí |
| Reddi 2020 | ISCA 2020 | MLPerf Inference | Varios | Cuatro escenarios de evaluación (resumen) | Justifica reportar percentiles y escenario single-stream | `reddi2020mlperf` | sí (5 de 47 autores) |

---

## D. Eficiencia de datos, preentrenamiento, fuga de datos y varianza

### Síntesis

**Qué se sabe.** Advertencia: las conclusiones que atribuyo aquí a Sun 2017 y He 2019 son las que recuerdo de esos artículos; en esta búsqueda solo verifiqué sus datos bibliográficos, no releí sus resúmenes, así que hay que confirmarlas antes de citarlas. Sobre el *escalado con datos*: Sun 2017 mostró que el rendimiento crece de forma aproximadamente logarítmica con el volumen de datos, y Mahmood 2022 que extrapolar una ley de potencias desde pocos puntos sobre o subestima fuertemente los datos necesarios, por lo que recomiendan varias rondas y funciones de ajuste más generales (abstract y página del proyecto). Sobre *preentrenamiento*: He 2019 mostró que entrenar desde cero alcanza al preentrenamiento con suficientes datos e iteraciones, pero que el preentrenamiento acelera la convergencia y ayuda con pocos datos; Wang 2023 compara preentrenamiento en ImageNet contra preentrenamiento en teledetección y encuentra ventajas complementarias. Wang 2020 (TFA) estableció que el ajuste fino simple es una línea base fuerte con pocos ejemplos y, lo que más interesa aquí, que **hay que muestrear varios grupos de ejemplos de entrenamiento** para que la evaluación sea estable.

Sobre *fuga de datos en video*: Figueiredo y Mendes 2024 demuestran con YOLOv8 que la partición aleatoria de frames correlacionados filtra información y proponen repartir por clústeres espaciotemporales; Glazner 2025 propone agrupar frames visualmente similares antes de dividir; y UTUAV (tema B) muestra el mismo efecto en video de dron en Latinoamérica. Sobre *varianza*: Bouthillier 2021 muestra que el muestreo de datos, la inicialización y los hiperparámetros aportan varianza comparable a las diferencias entre métodos y recomienda aleatorizar tantas fuentes como sea posible.

**Qué falta, y qué significa para el paper.** (1) La partición por clips ya **no es una novedad**: es buena práctica documentada desde 2024 y aplicada en UTUAV. Sirve como requisito, no como contribución. Lo que sí sería contribución es *cuantificar* la inflación (partición por frame contra partición por clip) en este dataset. (2) El diseño criticado por los revisores (un test distinto por subconjunto, una semilla) contradice directamente a Wang 2020, Mahmood 2022 y Bouthillier 2021. El diseño correcto es: **un único conjunto de prueba fijo, definido por clips; subconjuntos de entrenamiento anidados; al menos 3-5 repeticiones por tamaño variando semilla de submuestreo e inicialización; reporte de media, desviación e intervalo; ajuste de curva con función explícita**. (3) No encontré curvas de aprendizaje publicadas para detección OBB de vehículos desde dron, ni estudios de cuántos *clips* (no frames) se necesitan: el número efectivo de muestras independientes en video es un problema poco tratado.

### Tabla D

| Ref. corta | Venue | Método | Dataset | Métricas / hardware (fuente) | Relevancia y diferencia con nuestro trabajo | Clave BibTeX | Verificado |
|---|---|---|---|---|---|---|---|
| He 2019 | ICCV 2019 | Preentrenamiento vs desde cero | COCO | Sin cifras en lo leído | Justifica el experimento "DOTA-preentrenado vs COCO-preentrenado vs desde cero" | `he2019rethinking` | sí |
| Sun 2017 | ICCV 2017 | Escalado con 300 M de imágenes | JFT-300M | Crecimiento logarítmico (de memoria; contenido no releído) | Marco clásico para curvas de aprendizaje | `sun2017unreasonable` | sí (sin DOI de IEEE) |
| Mahmood 2022 | CVPR 2022 | Estimación de requerimientos de datos | Clasificación, detección, segmentación | Recomiendan ~5 rondas; factor de corrección (página del proyecto) | Sustento para ajustar y extrapolar la curva con cautela | `mahmood2022howmuchdata` | sí |
| Wang 2020 | ICML 2020 | TFA: ajuste fino en dos etapas | VOC, COCO, LVIS | +2 a +20 puntos sobre meta-aprendizaje (resumen) | Protocolo con múltiples muestras de entrenamiento | `wang2020tfa` | sí |
| Wang 2023 | IEEE TGRS 61 | Preentrenamiento en teledetección (MillionAID) | Varias tareas | Sin cifras confirmadas | Antecedente del efecto del dominio de preentrenamiento (DOTA en nuestro caso) | `wang2023rsp` | sí |
| **Figueiredo y Mendes 2024** | IEEE Access 12 | Partición por clústeres espaciotemporales | Video, YOLOv8 | Sin cifras en lo leído | **Respaldo directo de la partición por clips**; hay que citarlo | `figueiredo2024leakage` | sí |
| Glazner 2025 | arXiv 2511.13944 (aceptado en ICSEE 2026 según arXiv) | Selección por clústeres de embeddings | Datasets derivados de video | Sin cifras en lo leído | Alternativa cuando no hay identificador de clip | `glazner2025findleak` | sí |
| Bouthillier 2021 | MLSys 2021 | Modelo de varianza en benchmarks | 5 tareas | Estimador sesgado más barato (51x) (abstract) | Justifica semillas múltiples e intervalos | `bouthillier2021variance` | sí |

También relevante para D: `lepin2026utuav` (tema B) y `pineau2021reproducibility` (tema F).

---

## E. Conteo y seguimiento de vehículos en video aéreo

### Síntesis

**Qué se sabe.** El paradigma dominante es *tracking-by-detection*: SORT (Kalman + IoU), DeepSORT (añade apariencia; cerca de 45 % menos cambios de identidad, abstract), ByteTrack (asocia también detecciones de baja confianza), OC-SORT (corrige el ruido acumulado durante oclusiones) y BoT-SORT (compensación de movimiento de cámara; 80.5 MOTA, 80.2 IDF1 y 65.0 HOTA en MOT17, abstract). Ultralytics integra ByteTrack y BoT-SORT (según su documentación, no reverificada en esta búsqueda). Para dron, UAVMOT argumenta que el movimiento irregular de la cámara rompe el supuesto del filtro de Kalman. La métrica recomendada es HOTA, que equilibra detección y asociación. En conteo aéreo, Xiang 2018 reporta más de 90 % de exactitud con fondo fijo y 85 % con fondo móvil; Gellida-Coutino 2025 combina YOLOv8, ByteTrack y compensación con LoFTR, y reporta hasta 96.73 % en zonas sin obstrucción y 87.8 % en escenarios complejos sobre 27 horas de video (resumen de catálogo). Los datasets de trayectorias (DRIFT, CitySim, VRUD) usan OBB porque el rumbo y la huella exacta del vehículo mejoran velocidades y medidas de seguridad.

**Qué falta.** (1) Los trackers estándar asocian con IoU de cajas horizontales; no encontré una evaluación publicada del beneficio de asociar con IoU rotado o ProbIoU (ni verifiqué cómo asocia Ultralytics las cajas OBB en su modo de seguimiento; hay que revisarlo en el código). (2) No encontré evaluaciones de conteo por clase con las categorías de la región (mototaxi, combi). (3) No hay resultados de detección más seguimiento en una Raspberry Pi 4 para video aéreo. A 3.3 FPS, el supuesto de desplazamiento pequeño entre frames se debilita: hace falta medir cómo se degrada el conteo al bajar la tasa de cuadros. Es un experimento barato y, hasta donde encontré, no publicado.

### Tabla E

| Ref. corta | Venue | Método | Dataset | Métricas / hardware (fuente) | Relevancia y diferencia con nuestro trabajo | Clave BibTeX | Verificado |
|---|---|---|---|---|---|---|---|
| Bewley 2016 | ICIP 2016 | SORT | MOT | No aplica | Línea base mínima, apta para CPU | `bewley2016sort` | sí |
| Wojke 2017 | ICIP 2017 | DeepSORT | MOT16 | ~45 % menos cambios de identidad (abstract) | Requiere red de apariencia: costoso en la Pi | `wojke2017deepsort` | sí |
| Zhang 2022 | ECCV 2022 | ByteTrack | MOT17/20 | Sin cifras en lo leído | Tracker por defecto razonable (sin ReID) | `zhang2022bytetrack` | sí |
| Aharon 2022 | arXiv 2206.14651 | BoT-SORT | MOT17/20 | 80.5 MOTA, 80.2 IDF1, 65.0 HOTA en MOT17 (abstract) | Compensación de cámara útil si el dron deriva | `aharon2022botsort` | sí |
| Cao 2023 | CVPR 2023 | OC-SORT | MOT17/20 y otros | Sin cifras en lo leído | Robusto a oclusión (puentes, árboles) | `cao2023ocsort` | sí (sin DOI de IEEE) |
| Liu 2022 | CVPR 2022 | UAVMOT | VisDrone2019, UAVDT | Sin cifras en lo leído | Seguimiento específico para dron en movimiento | `liu2022uavmot` | sí |
| Luiten 2021 | IJCV 129(2) | HOTA | MOT | No aplica | Métrica a reportar si se añade seguimiento | `luiten2021hota` | sí |
| Xiang 2018 | Sensors 18(8) | Detección + seguimiento para conteo | Video aéreo | >90 % (fondo fijo), 85 % (fondo móvil) (abstract) | Antecedente clásico de conteo aéreo | `xiang2018aerialcounting` | sí |
| Gellida-Coutino 2025 | IEEE Access | YOLOv8 + ByteTrack + LoFTR | 27 h de video de dron (México, por confirmar) | 96.73 % / 87.8 % de exactitud de conteo (resumen de catálogo) | Conteo y velocidad desde dron con oclusión; HBB, sin borde | `gellidacoutino2025occluded` | parcial (sin DOI, volumen ni páginas) |

También relevantes para E: `lee2025drift`, `zheng2024citysim`, `salcedodelgado2026yucatan`, `wang2026vrud` (tema B).

---

## F. Metodología de evaluación

### Síntesis

**Qué se sabe.** Padilla 2021 documenta que "mAP" no es una sola métrica: cambia con la interpolación, los umbrales y el formato de anotación entre VOC, COCO y otros, y que hay que declarar la variante exacta. TIDE descompone el error en tipos (clasificación, localización, duplicados, fondo, omisiones) y es más informativo que una matriz de confusión para decidir dónde mejorar. Para cajas orientadas el punto crítico es el **cálculo del IoU**: el protocolo de DOTA usa IoU poligonal exacto, mientras que Ultralytics empareja predicciones y anotaciones en validación con ProbIoU, una similitud entre gaussianas. El propio código de Ultralytics advierte que su ruta de resultados fusionados puede dar un mAP algo menor que el script oficial por el cálculo con ProbIoU (comentario del validador, visto en un espejo del código; hay que verificar la versión usada). Dos efectos conocidos de la aproximación gaussiana: una caja casi cuadrada tiene una gaussiana casi isotrópica, de modo que **el error de ángulo deja de penalizarse**, y la relación entre ProbIoU e IoU poligonal no es lineal, así que los umbrales 0.50-0.95 no significan lo mismo en ambos.

En reproducibilidad, la lista de verificación de NeurIPS (Pineau 2021) y el análisis de varianza (Bouthillier 2021) son el estándar: código, datos, semillas, número de corridas, intervalos y descripción del cómputo.

**Qué falta, y riesgo para el paper.** No encontré ninguna comparación publicada entre mAP calculado con ProbIoU y con IoU poligonal sobre las mismas predicciones. Como los resultados del equipo (0.939 y 0.817) provienen del validador de Ultralytics, **no son directamente comparables con cifras de DOTA ni con detectores evaluados en MMRotate**. Para una comparación de arquitecturas justa hay que reevaluar todas las predicciones con un único evaluador de IoU poligonal. Reportar además la diferencia entre ambos evaluadores sería un aporte metodológico pequeño pero original, especialmente relevante con 81 % de cajas casi alineadas y vehículos de relación de aspecto variada.

### Tabla F

| Ref. corta | Venue | Método | Dataset | Métricas / hardware (fuente) | Relevancia y diferencia con nuestro trabajo | Clave BibTeX | Verificado |
|---|---|---|---|---|---|---|---|
| Padilla 2021 | Electronics 10(3) | Comparación de métricas + toolkit (15 métricas) | Varios | No aplica | Obliga a declarar la variante de mAP; trabajo latinoamericano (Brasil) | `padilla2021metrics` | sí |
| Bolya 2020 | ECCV 2020 | TIDE: descomposición de errores | COCO y otros | No aplica | Análisis de errores por clase (mototaxi vs moto, microbús vs minibús) | `bolya2020tide` | sí |
| Pineau 2021 | JMLR 22(164) | Programa de reproducibilidad de NeurIPS 2019 | No aplica | No aplica | Lista de verificación para la sección de reproducibilidad | `pineau2021reproducibility` | sí |

Reutilizadas en F: `llerena2021probiou`, `yang2021gwd`, `yang2021kld` (tema A); `bouthillier2021variance` (D); `luiten2021hota` (E); `reddi2020mlperf` (C).

---

## Huecos y posicionamiento

### Afirmaciones de novedad que el trabajo SÍ podría defender (con los experimentos indicados)

1. **Primer estudio de detección orientada de vehículos en vista cenital con la taxonomía del transporte urbano peruano** (9 clases, con mototaxi, microbús, minibús y bus articulado). Sustento: UTUAV usa 3 clases; YucaMex-Drone 8 categorías de otro contexto; el trabajo de Lima es a nivel de calle. *Redacción honesta*: "hasta donde sabemos" y limitada a literatura indexada. *Falta*: descripción formal del dataset (fuente, altura, licencia, estadísticas por clase), análisis de errores por clase (TIDE o matriz de confusión con intervalos) y confirmar con los organizadores que el dataset se puede describir y redistribuir.

2. **Evidencia controlada de OBB frente a HBB cuando la mayoría de las cajas está casi alineada con los ejes.** No encontré un estudio así. *Falta*: entrenar YOLO11n (HBB) y YOLO11n-OBB con los mismos datos, semillas y presupuesto; evaluar ambos con el mismo criterio (por ejemplo, convertir OBB a su HBB envolvente y también medir IoU poligonal); desagregar por ángulo (subconjunto de 81 % casi alineado contra 19 % rotado), por clase y por densidad; medir el costo en latencia de la cabeza OBB y de la NMS rotada en la Pi. Cualquiera de los dos resultados (OBB ayuda o no ayuda) es publicable.

3. **Caracterización rigurosa del despliegue OBB en CPU ARM de bajo costo.** No encontré mediciones de modelos OBB en Raspberry Pi, ni benchmarks de YOLO en Pi que reporten juntos percentiles, hilos, governor y temperatura. *Falta*: protocolo tipo Benoit-Cattin (corridas largas, con y sin ventilador, temperatura ambiente registrada, frecuencia de CPU y bandera de throttling), P50/P95/P99, barrido de hilos (1-4) y de resolución, desglose preproceso/inferencia/NMS, FP32 contra FP16 e INT8 en NCNN con su efecto en mAP, y al menos un segundo dispositivo (Raspberry Pi 5 y/o Jetson Orin Nano) y un segundo runtime (ONNX Runtime u OpenVINO/TFLite).

4. **Curva de aprendizaje con diseño estadístico correcto para ajuste fino OBB en video aéreo, medida en clips y no solo en frames.** *Falta*: test fijo por clips, subconjuntos anidados, 3-5 semillas por punto, intervalos de confianza, ajuste de curva (ley de potencias con término de saturación) y comparación de inicializaciones (DOTA-OBB, COCO, desde cero).

5. **Cuantificación de la inflación por fuga de datos en este dataset** (partición por frame contra partición por clip). Es una réplica, no un descubrimiento, pero aporta un número para un dominio nuevo y refuerza la credibilidad del resto.

6. **Diferencia entre mAP con ProbIoU y con IoU poligonal** sobre las mismas predicciones (aporte metodológico menor pero no encontrado en la literatura).

### Afirmaciones que NO podría defender

- **"La partición por clips es una contribución"**: es práctica documentada (Figueiredo y Mendes 2024; Glazner 2025) y ya aplicada en dron latinoamericano (UTUAV).
- **"Primer dataset o benchmark de tráfico desde dron en Latinoamérica"**: UTUAV (Colombia) y YucaMex-Drone (México) son anteriores.
- **"Estado del arte" o "arquitectura novedosa"**: no hay cambio arquitectónico y YOLO11 fue superado en su propia familia (YOLO12, YOLO26).
- **"Tiempo real en Raspberry Pi"**: 3.3 FPS no lo es bajo ninguna definición habitual; Rey 2025 concluye que ni la Pi 5 alcanza tiempo real con YOLOv8. Lo defendible es "procesamiento diferido o casi en línea de bajo costo", ligado a un caso de uso (aforo por intervalos).
- **"mAP comparable con DOTA u otros detectores"**: no mientras se evalúe con ProbIoU y sin un evaluador común.
- **"Los resultados generalizan a otras ciudades, alturas o condiciones"**: UTUAV reporta el cambio de altura como un problema de desplazamiento de dominio; sin evaluación cruzada no se puede afirmar.
- **"0.939 de mAP@50 demuestra que el problema está resuelto"**: con una sola semilla y un test posiblemente pequeño, el intervalo de confianza es desconocido.

### Experimentos mínimos para la versión de revista (por prioridad)

1. Comparación de arquitecturas con evaluador común de IoU poligonal: YOLO11n/s-OBB, YOLOv8n-OBB, YOLO12n-OBB, YOLO26n-OBB, RTMDet-R-tiny (MMRotate) y, si el cómputo alcanza, un detector de dos etapas como techo.
2. OBB contra HBB (afirmación 2).
3. Rediseño del escalado de datos (afirmación 4) con 3-5 semillas.
4. Benchmark de borde ampliado (afirmación 3): segundo dispositivo, segundo runtime, cuantización, protocolo térmico.
5. Fuga de datos cuantificada (afirmación 5).
6. Opcional pero de alto valor aplicado: conteo por clase con ByteTrack sobre OBB y sensibilidad a la tasa de cuadros (3, 5, 10, 30 FPS).

---

## Preguntas abiertas que la literatura no resuelve

1. ¿Cuánto aporta realmente OBB sobre HBB en escenas de tráfico cenital donde las calles forman una retícula y los vehículos siguen los ejes de la imagen? ¿Depende de la densidad (colas, paraderos) más que del ángulo?
2. ¿Qué tan distinto es el mAP con ProbIoU del mAP con IoU poligonal para vehículos pequeños y casi cuadrados (mototaxi visto desde arriba)? ¿Cambia el orden entre modelos?
3. ¿Cuál es la unidad efectiva de datos en video aéreo: frames, clips, intersecciones o minutos? ¿Cuántos clips independientes equivalen a 1 000 frames?
4. ¿El preentrenamiento en DOTA (satélite y aéreo de gran altura) aporta más que COCO para dron urbano a baja altura, o la diferencia desaparece con 1 000-3 000 frames?
5. ¿Cómo se degradan los detectores OBB bajo cuantización INT8? La regresión del ángulo podría ser más sensible que la de las cajas horizontales; no encontré evidencia en ningún sentido.
6. ¿Cuál es el costo de la NMS rotada en CPU ARM frente a la inferencia de la red? ¿Las cabezas sin NMS (YOLOv10, YOLO26) cambian el balance en una Pi?
7. ¿Qué tasa de cuadros mínima necesita un tracker para contar por clase en tráfico denso desde vista cenital? ¿Sirven 3 FPS?
8. ¿Cómo se comporta la latencia en la Pi 4B en el clima real de despliegue (Lima en verano, caja cerrada, sin ventilador)? Benoit-Cattin muestra que importa, pero para CNN de clasificación y en otro clima.
9. ¿Las clases regionales se confunden entre sí por apariencia o por tamaño? La frontera microbús/minibús/bus parece más una cuestión de escala métrica que de textura, lo que sugiere usar la altura de vuelo o la resolución en el suelo (GSD) como señal.
10. ¿Los resultados se transfieren entre ciudades latinoamericanas (Lima, Medellín, Mérida)? Existen ahora al menos tres datasets para intentar una evaluación cruzada, aunque con taxonomías distintas.

---

## Limitaciones de esta revisión

- No se leyó ningún artículo a texto completo; MDPI y DOAJ devolvieron error 403 al acceso automático, por lo que varias fuentes de MDPI se verificaron por Crossref, PubMed Central y resúmenes de búsqueda.
- SCImago bloqueó el acceso automático: ver las advertencias sobre cuartiles en `revistas.md`.
- La búsqueda fue en inglés y español; la literatura en chino y portugués está subrepresentada.
- La ausencia de un estudio ("no encontré") refleja unas 85 consultas y lecturas de páginas, no una revisión sistemática con protocolo PRISMA. Antes de escribir "no existe" en el paper conviene una búsqueda dirigida en Scopus y Web of Science.
- No pude localizar ningún artículo de IEEE Latin America Transactions de 2025-2026 comparable; el buscador no indexa bien esa revista.
