# 1. Definición del Problema de Visión Artificial

La fase de definición es el pilar de la arquitectura del sistema. Identificar correctamente la tarea de visión artificial determina la estructura del tensor de salida, la selección de la función de pérdida (*loss function*) y la estrategia de etiquetado de datos.

## 1.1. Categorización de Tareas Principales

En un entorno de producción, los problemas se clasifican según el nivel de granularidad requerido en la predicción:

| Tarea | Objetivo Técnico | Representación de Salida |
| :--- | :--- | :--- |
| **Clasificación de Imagen** | Asignar una o varias etiquetas a la imagen completa. | Vector de probabilidades ($P_k$) mediante *Softmax* o *Sigmoid*. |
| **Detección de Objetos** | Localizar y clasificar múltiples instancias dentro de una imagen. | Tuplas de $[x, y, w, h, class\_id, confidence]$ por cada objeto detectado. |
| **Segmentación Semántica** | Clasificar cada píxel de la imagen en una categoría. | Máscara de dimensiones $H \times W$ con valores de clase por píxel. |
| **Segmentación de Instancias** | Diferenciar entre distintos objetos de la misma clase a nivel de píxel. | Máscaras individuales para cada instancia detectada. |

## 1.2. Factores de Decisión y su Impacto en el Modelo

La selección de la tarea no solo depende del objetivo de negocio, sino de las restricciones técnicas y la complejidad del entorno:

1.  **Granularidad Espacial:**
    *   Si el contexto global es suficiente para la predicción, se opta por **Clasificación**. Requiere arquitecturas de codificación pura (*Backbones*) como ResNet o EfficientNet.
    *   Si la ubicación exacta es crítica (ej. guiado robótico), se requiere **Detección o Segmentación**. Esto obliga a utilizar arquitecturas con componentes adicionales como *Feature Pyramid Networks* (FPN) para manejar diferentes escalas.

2.  **Superposición de Objetos:**
    *   En escenarios donde los objetos se solapan, la detección por cajas (*Bounding Boxes*) puede ser ambigua. Aquí, la **Segmentación de Instancias** es necesaria para deslindar los bordes de cada entidad, aunque el costo computacional sea mayor.

3.  **Restricciones de Latencia:**
    *   La clasificación es la tarea menos costosa en términos de FLOPs (*Floating Point Operations*).
    *   La segmentación requiere procesar resoluciones de salida iguales a las de entrada, lo que incrementa significativamente el uso de memoria VRAM y el tiempo de inferencia.

## 1.3. Impacto en la Selección del Modelo (Model Selection)

La definición del problema dicta la topología de la red neuronal:

*   **Arquitecturas de Clasificación (Encoders):** Se enfocan en la reducción de dimensionalidad espacial para extraer características semánticas de alto nivel.
    *   *Decisión Técnica:* Elegir según el *trade-off* entre precisión y velocidad (ej. MobileNetV3 para dispositivos móviles vs. Vision Transformer (ViT) para máxima precisión en servidor).
*   **Arquitecturas de Detección (One-stage vs Two-stage):**
    *   *One-stage (YOLO, SSD):* Priorizan la velocidad realizando la regresión de cajas y clasificación en un solo paso.
    *   *Two-stage (Faster R-CNN):* Priorizan la precisión mediante una red de propuesta de regiones (RPN), ideal cuando los objetos son muy pequeños.
*   **Arquitecturas de Segmentación (Encoder-Decoder):**
    *   Utilizan una estructura en espejo (como U-Net) donde el *Decoder* recupera la resolución espacial perdida durante la contracción del *Encoder* mediante conexiones residuales (*Skip Connections*).
