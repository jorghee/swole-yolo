# Protocolo experimental (pre-registrado)

Este documento fija **qué se va a medir, cómo y qué se concluirá según el resultado**, antes de ejecutar los experimentos. Si algo cambia después, se anota la fecha y el motivo en la sección [Registro de cambios](#registro-de-cambios) (no se reescribe en silencio). Es la defensa más directa contra las críticas de las reseñas y contra el sesgo de confirmación.

Conceptos que se usan aquí están explicados en la [guía](../guia/index.md): [métricas](../guia/03-metricas.md), [datos y fuga](../guia/06-datos.md), [metodología](../guia/08-metodologia.md).

## 1. Hipótesis y reglas de decisión

Cada hipótesis indica su umbral de decisión **de antemano**. Todas las diferencias se juzgan con intervalos de confianza del 95 % por *bootstrap* sobre clips (§4).

| ID | Hipótesis | Se considera apoyada si… | Si no se apoya, se reporta… |
| --- | --- | --- | --- |
| **H1** | OBB supera a HBB en el subconjunto de vehículos **inclinados** (> 10° de un eje) | ΔAP@50–95 (OBB − HBB) en inclinados > 0 con IC95 que excluye 0 | "OBB no mejora medible; ventaja solo geométrica" |
| **H2** | En el conjunto **completo** la ventaja de OBB es pequeña (≤ 1 punto), porque ~81 % de las cajas están casi alineadas | IC95 de ΔAP@50–95 global contiene valores ≤ 1 punto | Se reporta el tamaño real |
| **H3** | Más **resolución** mejora a las clases pequeñas (moto, mototaxi) más que a las grandes | ΔAP de `motorcycle` entre 544 y 800–960 px > ΔAP de `car`, con IC95 | "La resolución no es el factor limitante" |
| **H4** | **Más clips (diversidad)** mejora más que **más frames por clip** (redundancia) con el mismo número de imágenes | AP(clip-diverso) − AP(frame-denso) > 0 con IC95 | "La diversidad no domina con estos datos" |
| **H5** | La partición **por frame** infla el mAP respecto a la partición **por clip**, y esta respecto a la partición **por escena** | AP(frame) > AP(clip) > AP(escena), con IC95 de al menos una de las diferencias que excluya 0 | Se reporta la ausencia de inflación |
| **H6** | El mAP medido con **ProbIoU** (Ultralytics) difiere del medido con **IoU poligonal**, y puede cambiar el orden entre modelos | \|ΔmAP\| > 1 punto en algún modelo, o cambia el ranking | "Los evaluadores concuerdan en este dominio" |
| **H7** | **INT8** da ≥ 1.5× de velocidad en la Pi con pérdida ≤ 2 puntos de mAP@50–95 | ambas condiciones se cumplen en la Pi 4 | Se reporta el compromiso real; si el export no lo permite, se declara |
| **H8** | La latencia en la Pi **degrada** > 10 % tras 30 min de carga continua sin ventilador | latencia media del último bloque de 5 min > 1.10 × la del primero | Se reporta que no hay degradación en ese entorno |

!!! warning "Un hallazgo negativo es un resultado, no un fracaso"
    Ninguna hipótesis puede "rescatarse" cambiando el criterio después. Si H1 no se apoya, el paper dice eso y mantiene su valor como estudio controlado.

## 2. Datos y particiones

### 2.1 Verificaciones previas (puerta G0.1)
1. Contar **clips**, **frames por clip** y confirmar si **todos** los frames están anotados.
2. Estimar **escenas/ubicaciones** (§2.2).
3. Registrar la **convención de anotación** (¿se omiten vehículos estacionados fuera de la calzada?) revisando ≥ 200 imágenes elegidas al azar con una semilla registrada.

### 2.2 Unidad de agrupación: la escena
El split por clip no impide que **dos clips del mismo cruce** queden en train y test. La unidad de agrupación será la **escena**:

1. Calcular un *embedding* visual de un frame de cada clip con un modelo de imagen preentrenado (p. ej., DINOv2).
2. Agrupar con *clustering* aglomerativo; fijar el umbral inspeccionando visualmente ejemplos de cada grupo (que un grupo ≈ una ubicación).
3. Guardar `clip → escena`.

### 2.3 Validación cruzada agrupada
- **5 pliegues** con `StratifiedGroupKFold` (grupos = escenas; estratificación aproximada por conteo de clase). Semilla fija y versionada: `data/splits/fold{0..4}.json`.
- En cada pliegue: **test** = el pliegue; **val** = ≈ 10 % de las escenas de entrenamiento (para elegir el checkpoint); **train** = el resto.
- El test de un pliegue **no se toca** hasta tener todas las configuraciones congeladas.
- Reportar **media y desviación entre pliegues** y el IC95 por *bootstrap* sobre clips (§4).

!!! note "Plan B (sin el dataset crudo)"
    Con solo los 3 000 frames de `data/`, se usa la misma CV agrupada sobre esos frames. El tamaño del conjunto de entrenamiento y la cobertura del test serán menores; se declara.

### 2.4 Muestreo de entrenamiento (para E1)
Para un tamaño \(N\) de imágenes de entrenamiento dentro de un pliegue:

| Estrategia | Definición |
| --- | --- |
| **clip-diverso** | 1 frame por clip, tantos clips como haga falta (máx. un frame por clip) |
| **frame-denso** | hasta \(k=5\) frames por clip, repartidos en el clip, sobre menos clips |

Tamaños: \(N\in\{250, 500, 1000, 2000, 4000\}\) (4 000 solo si el dataset crudo está disponible). Los subconjuntos están **anidados** dentro de cada estrategia.

### 2.5 Conjunto de entrenamiento estándar (E2–E6)
\(N_0=2000\) imágenes de entrenamiento, estrategia mixta (≈ 2 frames por clip), igual para todos los modelos del mismo pliegue.

## 3. Experimentos

### Configuración base común
Ultralytics (versión **fija y registrada**), `yolo11n-obb.pt` preentrenado en DOTA, batch 32, imgsz 640, 70 épocas, `patience=20`, `seed` = índice de pliegue, resto por defecto, **optimizador declarado explícitamente** (`optimizer` y `lr0` fijados; no `auto`). Entrenar en una sola máquina y registrar el entorno completo (§5).

| ID | Experimento | Variable | Controles |
| --- | --- | --- | --- |
| **E1** | Curva de aprendizaje | \(N\) y estrategia (§2.4) | modelo, imgsz, épocas, test |
| **E2** | OBB vs. HBB | tarea: `yolo11n-obb` vs. `yolo11n` (HBB) con etiquetas de caja envolvente; evaluación **común** (§4) | datos, pliegues, presupuesto, semilla |
| **E3** | Arquitecturas | `yolo11n/s-obb`, `yolov8n-obb`, `yolo26n-obb`; opcional `yolo12n-obb` (sin preentrenamiento DOTA, marcado) y RTMDet-R-tiny (MMRotate) | datos, pliegues, imgsz |
| **E4** | Resolución × épocas | imgsz ∈ {544, 640, 800, 960} × épocas ∈ {50, 100} (3 pliegues) | modelo, datos |
| **E5** | Preentrenamiento | DOTA-OBB / COCO-detect (backbone) / desde cero (150 épocas) | datos, imgsz |
| **E6** | Fuga | partición por frame / por clip / por escena con el mismo \(N_0\) y modelo | modelo, test (siempre por escena) |

!!! warning "E4 aísla un factor a la vez"
    La Tabla 4 del paper original compara 544 px/50 ép con 640 px/70 ép (dos factores). Aquí el diseño es **factorial** y además se entrena hasta convergencia (el original tenía el mejor checkpoint en las épocas 43–50 de 50).

### Etiquetas HBB para E2
La caja HBB de cada vehículo es la **envolvente** del rectángulo orientado. Se documenta y se prueba la conversión con casos conocidos (p. ej., el auto de 90×35 px a 45° ⇒ 88.4×88.4 px).

## 4. Métricas y estadística

### 4.1 Evaluador común
Un único evaluador, escrito y probado en este repositorio, aplicado a **todas** las predicciones (de Ultralytics, MMRotate, HBB convertidas, NCNN en la Pi):

| Variante | IoU usado | Para qué |
| --- | --- | --- |
| **Polígono** | intersección exacta de rectángulos rotados (estilo DOTA) | **métrica principal** y comparabilidad con la literatura |
| **ProbIoU** | el de Ultralytics | cuantificar la diferencia (H6) |
| **HBB** | caja horizontal envolvente | comparar HBB y OBB en igualdad de condiciones (E2) |

Pruebas unitarias obligatorias: cajas idénticas (IoU = 1), disjuntas (0), una caja girada 90° con lados intercambiados (misma caja), el caso de 45° del [capítulo 4](../guia/04-cajas-orientadas.md), y comparación con Ultralytics en cajas casi cuadradas.

### 4.2 Métricas
- **Principal:** AP@50–95 (polígono), promedio sobre clases.
- Secundarias: AP@50, AP@75, precisión y recall **a un umbral de confianza fijado en validación** (no el que maximiza F1 en test), AP por clase con su \(n\).
- **Estratificadas:** por **tamaño** (tercios del lado corto en píxeles originales), por **inclinación** (≤ 2°, 2–10°, > 10° de un eje), por **densidad** (objetos por imagen), por **clase**.
- **Conteo** (aplicación): error absoluto medio del número de vehículos por imagen, total y por clase.

### 4.3 Incertidumbre
- **Bootstrap por clip** (10 000 remuestreos de clips del test, reuniendo los pliegues): IC95 para cada métrica y para cada **diferencia pareada** entre configuraciones (mismos clips en ambos brazos).
- **Variación por inicialización:** 5 semillas sobre un pliegue para la configuración principal; reportar su desviación y comparar con la variación entre pliegues.
- Clases con \(n_{test}\) < 30: se marcan "exploratorias" y no entran en conclusiones.
- Comparaciones **pre-especificadas** (H1–H8); lo demás es exploratorio y se etiqueta así. Sin selección posterior de contrastes.

### 4.4 Qué se afirmará según el resultado (tabla de decisión)

| Resultado | Afirmación permitida |
| --- | --- |
| H1 apoyada, H2 apoyada | "OBB aporta en vehículos inclinados; en el conjunto completo el efecto es pequeño" |
| H1 no apoyada | "No se encontró ventaja medible de OBB sobre HBB en este dominio" |
| H3 apoyada | "La resolución es un factor crítico para vehículos pequeños" |
| H4 apoyada | "La diversidad de escenas importa más que la cantidad de frames" |
| H6 apoyada | "Los mAP de Ultralytics no son comparables con los de DOTA; reportamos ambos" |
| H7 / H8 | afirmaciones de despliegue **solo** para el dispositivo, SO y refrigeración medidos |

## 5. Entorno y registro (todo corrida)

Guardar automáticamente, para **cada** corrida: hash de commit, `pip freeze`, versión de Python/CUDA/driver, modelo de GPU, `args.yaml`, semilla, hora de inicio/fin, y el **log completo** del entrenamiento (en el original no se conservó, y por eso no se sabe qué optimizador eligió `auto`).

## 6. Protocolo de borde

### 6.1 Dispositivos y software
| Elemento | Requisito |
| --- | --- |
| Dispositivo principal | Raspberry Pi 4 B 8 GB (el del paper) |
| Segundo dispositivo | Raspberry Pi 5 **o** Jetson Orin Nano (S) |
| Sistema | imagen de SO fijada (versión, kernel), 64-bit |
| Versiones | `ncnn`, `ultralytics`, `onnxruntime`, Python; guardadas en el JSON |
| Refrigeración | condición declarada: **sin ventilador**, **ventilador pasivo/activo** (ambas) |
| Ambiente | temperatura ambiente registrada con termómetro |

### 6.2 Procedimiento por medición
1. Reposo de ≥ 15 min hasta temperatura de SoC estable y < 50 °C.
2. Governor fijado (`performance`) y también la configuración por defecto (se reportan ambas).
3. 5 imágenes de calentamiento y 300 de medición, batch 1; **mismo banco de imágenes** en todas las configuraciones.
4. Registrar por imagen: latencia end-to-end y desglose; cada 10 imágenes: temperatura, frecuencia, `vcgencmd get_throttled` (incluyendo los bits "ocurrió desde el arranque"), uso de CPU por núcleo.
5. Reportar **media, desviación, mediana, P95, P99, máximo**, throughput y detecciones por imagen.
6. **≥ 3 repeticiones** por configuración; reportar la dispersión entre repeticiones.

### 6.3 Matriz de configuraciones
| Eje | Valores |
| --- | --- |
| Modelo | YOLO11n-OBB (544 y 640 px), HBB equivalente, y el mejor de E3 |
| Runtime | NCNN, ONNX Runtime (CPU ARM) |
| Precisión | FP32, FP16; INT8 si el export lo permite (requiere conjunto de calibración fuera del test) |
| Hilos | 1, 2, 3, 4 (**verificar primero cómo fijarlos**: incertidumbre abierta desde el benchmark original) |
| Corrida larga | ≥ 30–60 min continuos, con y sin ventilador |

### 6.4 Precisión numérica en el dispositivo
Evaluar **mAP en la Pi** sobre el test completo del pliegue de referencia y compararlo con PyTorch en el servidor (mismo evaluador común). Reportar además la **equivalencia** de predicciones (IoU de emparejamiento entre salidas de PyTorch y NCNN para las mismas imágenes).

## 7. Aplicación: conteo (opcional)

1. **Por frame:** error de conteo por clase usando solo las anotaciones (no requiere seguimiento).
2. **Con seguimiento** (solo si el dataset crudo trae anotaciones en todos los frames de cada clip): construir pistas de referencia emparejando anotaciones entre frames consecutivos, evaluar ByteTrack sobre las detecciones y medir cómo cambia el conteo al submuestrear a 5 y 3 FPS (la tasa que alcanza la Pi).

## 8. Lista de verificación antes de ejecutar

- [ ] Puertas G0.1–G0.6 resueltas.
- [ ] Splits generados y **congelados** en `data/splits/`.
- [ ] Evaluador común con pruebas unitarias en verde.
- [ ] Ejecutor de experimentos con registro de entorno probado en una corrida piloto.
- [ ] Hipótesis y umbrales (§1) revisados por todo el equipo.
- [ ] Versión de Ultralytics y semillas fijadas.

## Registro de cambios

| Fecha | Cambio | Motivo |
| --- | --- | --- |
| 2026-11-11 | Versión inicial | Plan de mejora (Etapa 5) |
