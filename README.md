# Detección y Clasificación Vehicular con OBB mediante Fine-Tuning de YOLOv8

Proyecto de visión artificial desarrollado para el **SMART Challenge 2026: IA para la Movilidad del Perú (MTC)**. Enfocado en el análisis de tráfico urbano mediante el procesamiento de secuencias de video a 10 FPS y la detección precisa de vehículos utilizando cajas delimitadoras orientadas (Oriented Bounding Boxes - OBB).

## Propuesta Técnica

Desarrollar un modelo de visión artificial basado en la arquitectura **YOLOv8-OBB** aplicando *Transfer Learning* (Fine-Tuning) sobre pesos pre-entrenados. Dado el volumen masivo del dataset (aprox. 50 GB), la estrategia central prioriza la eficiencia matemática y computacional. 

En lugar de entrenar desde cero, se congelarán las capas iniciales de extracción de características, enfocando el entrenamiento en la adaptación de la red a las condiciones de iluminación, ruido y tipología vehicular específicas de las calles analizadas. El pipeline de datos aprovechará el paralelismo mediante múltiples *workers* de CPU para saturar eficientemente la GPU desde un almacenamiento NVMe de alto ancho de banda, garantizando un flujo continuo durante el entrenamiento sin cuellos de botella de I/O.

---

## Estado del Arte (State of the Art)

Para la detección de objetos con cajas delimitadoras orientadas (OBB), la literatura actual aborda el problema de la rotación ($\theta$) dividiendo las arquitecturas en dos enfoques principales. La selección de nuestro modelo se fundamenta en un análisis de *trade-off* entre la precisión geométrica (mAP) y la latencia computacional (FPS) requerida para procesar secuencias de video:

* **Modelos de Dos Etapas (Two-Stage Detectors - ej. Oriented R-CNN, RoI Transformer):** Históricamente, han sido el estándar de oro para tareas complejas con cajas rotadas. Utilizan una red extractora de características combinada con una RPN (*Region Proposal Network*) que primero genera múltiples "propuestas" de dónde podría haber un objeto, para luego aplicar una regresión fina sobre los ángulos. 
    * **Limitación en nuestro contexto:** Su alta densidad paramétrica y las múltiples pasadas por el grafo de computación generan un cuello de botella en la VRAM. Son ineficientes para alcanzar los 10 FPS continuos necesarios en el análisis de tráfico urbano, especialmente bajo hardware con memoria gráfica restringida.
* **Modelos de Una Etapa (Single-Stage Detectors - ej. arquitecturas basadas en YOLO-OBB):** Representan el estado del arte actual para despliegues de alto rendimiento. Las versiones modernas de YOLO emplean un diseño *Anchor-free* (sin cajas predefinidas), prediciendo directamente el centroide, las dimensiones y el ángulo de inclinación en un solo tensor de salida mediante funciones de pérdida probabilísticas.
    * **Ventaja competitiva:** Al procesar la imagen en una única pasada (Single-Shot), reducen drásticamente la latencia de inferencia y la huella en memoria gráfica (VRAM). Esto nos permite procesar el dataset masivo (50 GB) e iterar en el *Fine-Tuning* maximizando el tamaño del *batch* en un entorno local, manteniendo una precisión geométrica altamente competitiva frente a los modelos de dos etapas.

---

## Stack Tecnológico

Es fundamental alinear el hardware y el software con las exigencias de procesamiento del dataset completo.

| Componente | Tecnología | Justificación Técnica |
| :--- | :--- | :--- |
| **Lenguaje** | Python 3.10+ | Estándar de la industria con soporte nativo para librerías de tensores y visión. |
| **Framework de ML** | PyTorch 2.x | Base para Ultralytics; ofrece control granular sobre el grafo de computación y memoria VRAM. |
| **Motor de Detección** | Ultralytics (YOLOv8) | Implementación SOTA de OBB con utilidades optimizadas para entrenamiento distribuido. |
| **Gestión de Datos** | FiftyOne | Visualización y curación de datasets; permite identificar errores de etiquetado en OBB de forma eficiente. |
| **Procesamiento de Video** | OpenCV / PyAV | Decodificación rápida de frames minimizando el overhead de CPU. |
| **Experiment Tracking** | Weights & Biases (W&B) | Registro de curvas de pérdida, métricas Macro AP y versionamiento de modelos. |
| **Hardware Mínimo** | NVIDIA GPU (min. 24GB VRAM) | Necesario para manejar arquitecturas Large con batch sizes aceptables en 50 GB de datos.* |

*\*Nota: Para el desarrollo local, pruebas de concepto (PoC) y arquitecturas Nano/Small, el pipeline está optimizado para ejecutarse en entornos con 8GB de VRAM.*

---

## Resultados Preliminares: Prueba de Concepto (PoC)

Como primer hito del proyecto, se realizó un Análisis Exploratorio de Datos (EDA) y la validación del pipeline de transformación de anotaciones utilizando una muestra representativa de 300 imágenes extraídas del dataset original.

**El Problema de Ingesta:**
Los modelos *Single-Stage* como YOLO-OBB requieren que las anotaciones de entrenamiento se definan estrictamente mediante las coordenadas de los cuatro vértices del polígono $(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4)$. Sin embargo, el dataset crudo del MTC proporciona vectores estáticos en formato tabular (CSV) basados en el centroide y el ángulo $(x_c, y_c, w, h, \theta)$. 

**Resultados obtenidos:**
1.  **Transformación Geométrica:** Se implementó y validó el algoritmo de conversión en Python mediante la aplicación de matrices de rotación utilizando las primitivas matemáticas de OpenCV.
2.  **Validación Visual:** El script decodifica exitosamente la fila del CSV, calcula el polígono y proyecta la máscara resultante sobre el fotograma original sin distorsión de perspectiva.
    
    *(Nota: Reemplazar el siguiente enlace con la ruta real de la imagen generada en Jupyter)*
    `![Validación Visual OBB - Jupyter Notebook](./ruta/a/tu/imagen_generada.jpg)`

3.  **Conclusión:** El motor de transformación matemática y el pipeline de lectura de datos están validados. El sistema se encuentra arquitectónicamente listo para escalar al multiprocesamiento paralelo del dataset completo de 50 GB e iniciar la fase de Fine-Tuning del modelo base.