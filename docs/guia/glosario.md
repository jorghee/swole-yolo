# Glosario

Orden alfabético. Entre paréntesis, el capítulo donde se explica a fondo.

**Ablación** (8)
:   Experimento que quita o cambia un solo componente para medir cuánto aporta.

**Activación (función de)** (1)
:   Función no lineal aplicada a la salida de una neurona (aquí SiLU).

**Anchor** (2)
:   Caja de referencia de tamaño fijo usada por detectores antiguos. YOLO11 es *anchor-free*.

**AP / mAP** (3)
:   *Average Precision*: área bajo la curva precisión–recall de una clase. mAP: promedio entre clases.

**Backbone / Neck / Head** (5)
:   Partes de un detector: extracción de características / mezcla de escalas / predicción final.

**Batch** (1)
:   Grupo de muestras procesadas juntas en un paso de entrenamiento (32 en el paper).

**BatchNorm** (5)
:   Normaliza las activaciones de una capa para estabilizar el entrenamiento. Se fusiona con la convolución al inferir.

**Bootstrap** (3, 8)
:   Remuestreo con reemplazo para estimar la incertidumbre de una métrica.

**Calentamiento (warm-up)** (7)
:   Ejecuciones iniciales descartadas al medir tiempos.

**Clip** (6)
:   Un video del que se extraen varios frames; unidad de agrupación para evitar fuga.

**Confianza (score)** (2)
:   Número en [0,1] con que el modelo valora una detección.

**Confusor** (8)
:   Factor que cambia junto con la variable de interés y contamina la conclusión.

**Convolución / kernel** (1)
:   Filtro local de pesos que se desliza sobre la imagen.

**DFL** (2, 5)
:   *Distribution Focal Loss*: pérdida para predecir bordes como distribución.

**DOTA** (4)
:   Benchmark de imágenes aéreas con cajas orientadas; origen del preentrenamiento.

**Edge (borde)** (7)
:   Ejecutar el modelo en el propio dispositivo, no en un servidor.

**Época (epoch)** (1)
:   Una pasada completa por los datos de entrenamiento.

**Fine-tuning** (5)
:   Continuar entrenando un modelo preentrenado con datos propios.

**FLOPs** (1)
:   Operaciones de punto flotante necesarias por predicción; medida de coste independiente del hardware.

**FN / FP / TP** (3)
:   Falso negativo / falso positivo / verdadero positivo.

**FPS** (7)
:   Imágenes por segundo; \(1000/\text{latencia en ms}\) con batch 1.

**Fuga de datos (leakage)** (6)
:   Información de test que influye en el entrenamiento.

**Governor (CPU)** (7)
:   Política que regula la frecuencia de la CPU (`performance`, `ondemand`…).

**Gradiente / backpropagation** (1)
:   Derivada de la pérdida respecto a cada parámetro / algoritmo para calcularla.

**HBB** (2, 4)
:   Caja horizontal (*horizontal bounding box*).

**Hiperparámetro** (1)
:   Valor elegido antes de entrenar (batch, épocas, resolución…).

**IoU** (2)
:   Intersección sobre unión entre dos cajas.

**Latencia / P95** (7)
:   Tiempo por imagen / valor que el 95 % de las mediciones no supera.

**Letterbox** (1)
:   Redimensionar conservando proporción y rellenando con gris.

**Loss (pérdida)** (1, 2)
:   Número que mide cuán mal predice el modelo; se minimiza al entrenar.

**Matriz de confusión** (3)
:   Tabla clase real × clase predicha (con *background*).

**Mosaic** (5)
:   Aumento de datos que combina 4 imágenes en una.

**NCNN** (7)
:   Motor de inferencia ligero (Tencent) optimizado para ARM/móviles.

**NMS** (2)
:   *Non-Maximum Suppression*: elimina detecciones duplicadas.

**OBB** (4)
:   *Oriented Bounding Box*: caja con ángulo de rotación.

**ONNX** (7)
:   Formato intermedio de intercambio de modelos.

**Overfitting (sobreajuste)** (1)
:   Memorizar el entrenamiento y fallar en datos nuevos.

**Parámetro** (1)
:   Valor interno del modelo que el entrenamiento ajusta (2.66 M en YOLO11n-OBB).

**Precisión / Recall** (3)
:   \(TP/(TP+FP)\) / \(TP/(TP+FN)\).

**ProbIoU** (4)
:   IoU probabilístico entre cajas orientadas representadas como gaussianas.

**Stride** (1, 2)
:   Factor de reducción de resolución de un mapa de características (8, 16, 32).

**Split** (1, 6)
:   Partición de los datos en train, validación y test.

**TAL** (2, 5)
:   *Task-Aligned Assigner*: regla que asigna candidatas a objetos reales al entrenar.

**Throttling (térmico)** (7)
:   Reducción automática de frecuencia por temperatura alta.

**Transfer learning** (5)
:   Reaprovechar un modelo entrenado en otra tarea como punto de partida.

**YOLO** (2, 5)
:   *You Only Look Once*: familia de detectores de una etapa.
