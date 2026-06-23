## Stack tecnológico

Es fundamental incluir esta sección para garantizar que el hardware y el software estén alineados con las exigencias de procesamiento de 50 GB aprox. de datos.

| Componente | Tecnología | Justificación Técnica |
| :--- | :--- | :--- |
| **Lenguaje** | Python 3.10+ | Estándar de la industria con soporte nativo para librerías de tensores y visión. |
| **Framework de ML** | PyTorch 2.x | Base para Ultralytics; ofrece control granular sobre el grafo de computación y memoria VRAM. |
| **Motor de Detección** | Ultralytics (YOLOv8) | Implementación SOTA de OBB con utilidades optimizadas para entrenamiento distribuido. |
| **Gestión de Datos** | FiftyOne | Visualización y curación de datasets; permite identificar errores de etiquetado en OBB de forma eficiente. |
| **Procesamiento de Video** | OpenCV / PyAV | Decodificación rápida de frames minimizando el overhead de CPU. |
| **Experiment Tracking** | Weights & Biases (W&B) | Registro de curvas de pérdida, métricas Macro AP y versionamiento de modelos. |
| **Hardware Mínimo** | NVIDIA GPU (min. 24GB VRAM) | Necesario para manejar arquitecturas Large con batch sizes aceptables en 50 GB de datos. |

