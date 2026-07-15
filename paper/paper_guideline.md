# Borrador y guía de resultados — YOLO11n-OBB para tráfico peruano

> **Estado del documento (14 de julio de 2026).** Este archivo contiene texto
> listo para adaptar a la plantilla LNCS y resultados ya medidos en el conjunto
> de prueba. Las mediciones en Raspberry Pi siguen pendientes y aparecen como
> `TBD`; no deben sustituirse por las métricas de CPU de escritorio.

## Título propuesto

*Fine-Tuning YOLO11n-OBB for Oriented Vehicle Detection in Peruvian Traffic
Scenes: Data Scaling and Edge Deployment on Raspberry Pi*

**Keywords:** oriented object detection; vehicle detection; YOLO11n-OBB; edge
AI; Raspberry Pi; intelligent transportation systems.

## Nota metodológica esencial

El estudio contiene dos familias de experimentos:

| ID | Imágenes | Épocas | Tamaño de entrada registrado | Propósito |
|---|---:|---:|---:|---|
| F50-1000 | 1000 | 50 | 544 (efectivo) | Escalamiento de datos |
| F50-1500 | 1500 | 50 | 544 (efectivo) | Escalamiento de datos |
| F50-2000 | 2000 | 50 | 544 (efectivo) | Escalamiento de datos |
| F50-2500 | 2500 | 50 | 544 (efectivo) | Escalamiento de datos y referencia |
| F70-2500 | 2500 | 70 | 640 | Modelo final candidato para edge |

Los archivos `args.yaml` y `result_dataset_*.json` de las corridas de 50
épocas registran el valor solicitado `imgsz: 540`. YOLO ajusta el tamaño de
entrada al múltiplo de *stride* 544, que es la resolución efectiva reportada
en el manuscrito. Esta distinción debe conservarse para trazabilidad.

No se reporta ni se discute el subconjunto de **3000 imágenes**: su distribución
está desproporcionada y no forma parte de la evidencia experimental del paper.

La comparación F50-1000 a F50-2500 sí permite analizar el tamaño del conjunto,
porque mantiene fijos el modelo, las épocas, el tamaño de entrada efectivo y el
batch.
F70-2500 comparte los datos con F50-2500, pero cambia simultáneamente las
épocas y la resolución; por ello sirve para seleccionar un candidato final,
**no** para atribuir la mejora exclusivamente al número de imágenes.

---

## Resumen (versión de trabajo, en inglés)

Vehicle monitoring in resource-constrained urban areas requires detectors that
are both accurate under oblique viewpoints and practical on low-cost hardware.
We fine-tune YOLO11n-OBB, initialized from a DOTA-pretrained checkpoint, for
nine vehicle categories in Peruvian traffic imagery. To study data efficiency,
we construct clip-level train, validation, and test partitions for subsets of
1,000, 1,500, 2,000, and 2,500 images, thereby avoiding leakage among adjacent
video frames. Under a fixed 50-epoch, 544-pixel effective-input protocol, mAP@50 increases from
66.2% with 1,000 images to 91.4% with 2,500 images; mAP@50--95 rises from
54.8% to 78.5%. A second 2,500-image configuration trained for 70 epochs at
640 pixels reaches 93.9% mAP@50 and 81.7% mAP@50--95 with 2.66 million
parameters. Desktop CPU measurements are reported only as a four-thread proxy;
the exported NCNN model will be benchmarked on a Raspberry Pi using a
predefined protocol. These results indicate that a compact oriented detector
can attain high accuracy with a modest, domain-specific dataset while providing
a reproducible path toward edge deployment for traffic monitoring in Peru.

**Importante:** reemplazar la última oración del resumen por un resultado de
Raspberry Pi solo después de completar la tabla de la sección 6.3. No afirmar
todavía que es tiempo real en la Pi, ni que hubo cuantización, porque ello no
ha sido medido/documentado en los artefactos actuales.

---

## 1. Introducción

El monitoreo automático del tráfico puede apoyar la planificación urbana, el
conteo vehicular y la operación de intersecciones. Sin embargo, muchas
municipalidades requieren soluciones que funcionen con infraestructura de bajo
costo y que reconozcan la diversidad del parque automotor local. Las cámaras
de vigilancia elevadas producen vehículos rotados respecto a los ejes de la
imagen; en escenas congestionadas, una caja horizontal incluye demasiado fondo
o se superpone con objetos vecinos.

Las cajas delimitadoras orientadas (OBB) representan explícitamente la
orientación del vehículo y son adecuadas para esta geometría. No obstante, su
adaptación a escenas de tráfico peruanas presenta dos retos: la disponibilidad
limitada de anotaciones específicas del dominio y la necesidad de desplegar el
modelo sin depender de una GPU. Este trabajo estudia el fine-tuning de
YOLO11n-OBB sobre nueve categorías vehiculares y separa rigurosamente los clips
de video entre entrenamiento, validación y prueba.

Las contribuciones del artículo son las siguientes:

- Un protocolo reproducible para convertir anotaciones orientadas de nueve
  clases vehiculares de escenas peruanas al formato YOLO-OBB y dividir los
  datos por clips completos.
- Un estudio de escalamiento con 1,000--2,500 imágenes bajo una configuración
  fija de 50 épocas y 544 píxeles efectivos, evaluado una sola vez en el conjunto de
  prueba de cada subconjunto.
- Un modelo candidato de mayor capacidad de entrada, entrenado con 2,500
  imágenes, 70 épocas y 640 píxeles, que alcanza 93.9% mAP@50 y 81.7%
  mAP@50--95.
- Un protocolo y una tabla predefinidos para medir en una Raspberry Pi el
  export NCNN del modelo, evitando confundir resultados de escritorio con
  resultados de hardware embebido.

El resto del artículo revisa el trabajo relacionado, describe los datos y el
protocolo de entrenamiento, presenta los resultados y detalla las limitaciones
y el plan de evaluación en edge.

## 2. Trabajo relacionado — puntos que deben citarse

La versión final debe incluir referencias primarias para: (i) DOTA y detección
de objetos orientados; (ii) la documentación/publicación de YOLO11-OBB usada
en el experimento; (iii) detección de vehículos con OBB en cámaras elevadas;
y (iv) inferencia NCNN o en CPU ARM. Evitar afirmar que ningún dataset público
contiene ciertas clases sin una revisión bibliográfica que lo sustente.

La discusión debe diferenciar dos aportes: la adaptación al dominio vehicular
peruano y la evaluación empírica de una configuración nano. La Raspberry Pi es
un objetivo experimental pendiente, no un resultado de la revisión de
literatura ni una contribución ya demostrada.

## 3. Datos y preprocesamiento

Las imágenes proceden de videos de tráfico y contienen las clases `car`,
`van`, `microbus`, `minibus`, `bus`, `articulated_bus`, `truck`, `mototaxi` y
`motorcycle`. Las anotaciones de entrada tienen la forma
`(class, cx, cy, w, h, angle)`. Para YOLO-OBB, cada caja se convierte en cuatro
vértices normalizados:

\[
\begin{aligned}
x_k &= c_x + d_{x,k}\cos\theta - d_{y,k}\sin\theta,\\
y_k &= c_y + d_{x,k}\sin\theta + d_{y,k}\cos\theta,
\end{aligned}
\]

donde \((d_{x,k}, d_{y,k})\) son las cuatro combinaciones de
\((\pm w/2, \pm h/2)\). Las coordenadas se dividen por el ancho o alto de la
imagen y se acotan al intervalo \([0,1]\).

Para formar cada subconjunto, se asigna una cuota por video y se seleccionan
frames temporalmente espaciados mediante `np.linspace`. Después, las imágenes
se agrupan por `clip_id` y un clip se asigna íntegramente a train, validation o
test. Las semillas son `42 + dataset_size`; por tanto, los experimentos usan
las semillas 1042, 1542, 2042 y 2542. Este diseño reduce la fuga temporal que
ocurriría si frames vecinos aparecieran en particiones distintas.

| Subconjunto | Train | Validation | Test | Total |
|---:|---:|---:|---:|---:|
| 1000 | 700 | 150 | 150 | 1000 |
| 1500 | 1050 | 225 | 225 | 1500 |
| 2000 | 1400 | 300 | 300 | 2000 |
| 2500 | 1748 | 376 | 376 | 2500 |

**Pendiente antes de envío:** agregar una tabla de instancias por clase y una
figura de ejemplos anotados. No inferir el desbalance de clases sin contar las
anotaciones de los subconjuntos finales.

## 4. Metodología experimental

El modelo base es `yolo11n-obb.pt`, con 2,655,478 parámetros. Se parte de sus
pesos preentrenados y se selecciona el mejor checkpoint durante el entrenamiento.
Cada mejor checkpoint se evalúa una única vez sobre `split=test`.

Los parámetros comunes registrados son batch size 32, paciencia 20, optimizador
automático de Ultralytics y semillas deterministas. En la configuración de 70
épocas/640 píxeles, los hiperparámetros registrados incluyen `lr0=0.01`,
`momentum=0.937`, `weight_decay=0.0005`, `mosaic=1.0`, `mixup=0.0` y
`fliplr=0.5`. Para no sobredeclarar resultados, la versión final debe reportar
estos ajustes como configuración del software y citar la versión exacta de
Ultralytics, PyTorch, CUDA, GPU y sistema operativo utilizada.

Se reportan precision, recall, mAP@50 y mAP@50--95 para OBB. La latencia de
escritorio se midió sobre 30 imágenes, con batch 1 y cuatro hilos de CPU. Es un
*proxy* de ingeniería: no equivale a un benchmark de Raspberry Pi.

## 5. Resultados de detección

### 5.1 Escalamiento con protocolo fijo: 50 épocas y 544 píxeles efectivos

| Imágenes | mAP@50 | mAP@50--95 | Precision | Recall | Latencia CPU (ms/imagen) | FPS CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.6622 | 0.5483 | 0.7742 | 0.6425 | 30.99 | 32.27 |
| 1500 | 0.8167 | 0.6769 | 0.8352 | 0.7058 | 32.11 | 31.14 |
| 2000 | 0.8129 | 0.6873 | 0.8616 | 0.7219 | 32.98 | 30.32 |
| 2500 | **0.9140** | **0.7847** | **0.9053** | **0.8517** | 32.89 | 30.41 |

Entre 1000 y 2500 imágenes, mAP@50 aumenta 25.2 puntos porcentuales y
mAP@50--95 aumenta 23.6 puntos. La mejora no es estrictamente monótona entre
1500 y 2000 imágenes (mAP@50 baja 0.4 puntos), lo que debe reportarse tal cual:
puede reflejar la composición de los clips y el tamaño finito de los conjuntos
de prueba. El resultado de 2500 imágenes es el mejor dentro de esta familia
controlada.

**Figura sugerida:** curva con dos series (mAP@50 y mAP@50--95) y eje x igual
a {1000, 1500, 2000, 2500}. La leyenda debe decir “50 epochs, 544 px effective
input”. No
incluir 3000 imágenes ni mezclar en esta curva el experimento F70-2500.

### 5.2 Configuración final candidata: 2500 imágenes, 70 épocas y 640 píxeles

| Configuración | mAP@50 | mAP@50--95 | Precision | Recall | Parámetros | Tamaño de pesos |
|---|---:|---:|---:|---:|---:|---:|
| F50-2500 (50 ep, 544 px efectivo) | 0.9140 | 0.7847 | 0.9053 | 0.8517 | 2,655,478 | 5.54 MB |
| F70-2500 (70 ep, 640 px) | **0.9394** | **0.8175** | **0.9281** | **0.8595** | 2,655,478 | 5.60 MB |

En el mismo subconjunto de 2500 imágenes, F70-2500 supera a F50-2500 en 2.5
puntos de mAP@50 y 3.3 puntos de mAP@50--95. Como épocas y tamaño de entrada
cambian a la vez, el artículo debe describir esta diferencia como una
comparación de configuraciones, no como una ablación causal de la resolución o
del número de épocas.

Para F70-2500, los AP@50 por clase son:

| Clase | AP@50 |
|---|---:|
| car | 0.993 |
| van | 0.962 |
| microbus | 0.924 |
| minibus | 0.941 |
| bus | 0.995 |
| articulated_bus | 0.995 |
| truck | 0.981 |
| mototaxi | 0.915 |
| motorcycle | 0.749 |

`motorcycle` presenta el AP@50 más bajo. Es apropiado plantear como hipótesis
la oclusión, el tamaño pequeño y la variación visual, pero la causa no queda
demostrada sin análisis de errores y distribución por clase.

## 6. Evaluación pendiente en Raspberry Pi

El modelo a evaluar es **F70-2500**, usando el export NCNN ya generado. Esta
sección está intencionalmente preparada para registrar la medición real.

### 6.1 Información que debe acompañar al benchmark

- Modelo de Raspberry Pi, RAM, sistema operativo de 64 bits y versión de
  kernel.
- Frecuencia/estado térmico de CPU, método de alimentación y si hubo ventilación.
- Versión de NCNN y método de ejecución (API, binario o wrapper).
- Hilos de CPU, número de imágenes, calentamiento previo y si el tiempo incluye
  preprocesamiento, postprocesamiento y escritura de salida.
- Resolución de inferencia y si se usó FP32, FP16 o INT8. No llamar
  “cuantizado” al modelo salvo que se documente una conversión/calibración
  cuantizada.

### 6.2 Tabla de resultados por completar

| Formato / dispositivo | Resolución | Hilos | Imágenes | Latencia media (ms) | P50 / P95 (ms) | FPS | mAP@50 | mAP@50--95 | Estado |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch, CPU de escritorio (*proxy*) | 640 | 4 | 30 | 33.78 | TBD | 29.60 | 0.9394 | 0.8175 | Medido |
| NCNN, Raspberry Pi | 640 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pendiente |
| NCNN, Raspberry Pi | 544 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pendiente |
| NCNN, Raspberry Pi | 416 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pendiente |
| NCNN, Raspberry Pi | 320 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pendiente |

Para cada resolución, evaluar exactamente el mismo test de 376 imágenes del
subconjunto de 2500 cuando sea viable. Si se mide mAP a otra resolución,
documentar el preprocesamiento y la herramienta de evaluación. Si solo se mide
latencia, marcar las columnas de mAP como “no evaluado”, no estimarlas.

**Párrafo listo para completar tras las pruebas:**

> On the Raspberry Pi [MODEL], the NCNN export of F70-2500 achieved [FPS] FPS
> at [RESOLUTION] pixels, with a mean latency of [LATENCY] ms per image. The
> measurement used [THREADS] CPU threads over [N] images after [WARMUP]
> warm-up runs. At this resolution, the test accuracy was [MAP50] mAP@50 and
> [MAP5095] mAP@50--95 [or: accuracy was not re-evaluated on device].

## 7. Discusión y limitaciones

El resultado controlado con 2500 imágenes muestra que un detector OBB nano
puede alcanzar alta precisión en este dominio. No obstante, cada tamaño de
subconjunto tiene su propia partición de clips y conjunto de prueba; por ello
la curva de escalamiento combina el efecto de más datos con variación natural
de las particiones. Una extensión más fuerte del estudio evaluaría todos los
modelos contra un único test fijo, o repetiría las particiones con varias
semillas y reportaría intervalos de confianza.

Tampoco se han completado pruebas en Raspberry Pi. Las cifras de 30--32 FPS
son mediciones de CPU de escritorio limitadas a cuatro hilos y solo sirven como
referencia preliminar. La viabilidad en edge y cualquier compromiso entre
resolución, exactitud y latencia deben concluirse únicamente cuando se añadan
los resultados de la sección 6.

Entre las limitaciones adicionales a documentar están la cobertura geográfica,
las condiciones de iluminación/clima, la distribución por clase y la ausencia
actual de tracking para conteo o velocidad. El trabajo futuro puede incorporar
escenas nocturnas, más cámaras, seguimiento multiobjeto y una evaluación
energética en campo.

## 8. Conclusión (versión previa a Raspberry Pi)

We showed that fine-tuning YOLO11n-OBB on clip-level partitions of Peruvian
traffic imagery yields 91.4% mAP@50 under a controlled 50-epoch, 544-pixel
protocol with 2,500 images. A 70-epoch, 640-pixel configuration on the same
subset reached 93.9% mAP@50 and 81.7% mAP@50--95. These findings motivate an
on-device NCNN evaluation, whose results will determine the final claims about
real-time edge deployment.

Tras completar la Raspberry Pi, añadir la cifra real de la mejor configuración
y actualizar el resumen, las contribuciones y esta conclusión de forma
consistente.

## Lista de verificación antes de enviar

- [ ] Añadir referencias verificadas y formateadas en la plantilla objetivo.
- [ ] Registrar hardware, versiones de software y GPU de entrenamiento.
- [ ] Añadir distribución de instancias por clase y ejemplos de anotaciones.
- [ ] Generar la figura de escalamiento solo con F50-1000 a F50-2500.
- [ ] Ejecutar y documentar el benchmark NCNN en Raspberry Pi.
- [ ] Reemplazar todos los `TBD` únicamente con valores observados.
- [ ] Comprobar que no aparezca ningún resultado ni interpretación de
      `dataset_3000` en el manuscrito, tablas, figuras o conclusiones.
