# 2. Selección del Modelo Base: YOLOv8-OBB (Oriented Bounding Boxes)

Para este sistema de detección vehicular a gran escala, se selecciona la arquitectura **YOLOv8-OBB** (específicamente la versión **Large** o **X-Large** dado el volumen de 50 GB aproximadamente). Esta elección se fundamenta en la necesidad de predecir cajas delimitadoras con orientación ($cx, cy, w, h, \theta$) de forma nativa, optimizando la precisión en entornos urbanos densos y tomas con perspectiva.

## 2.1. Justificación Técnica de la Arquitectura

YOLOv8-OBB es un detector de una sola etapa (*one-stage*) que integra la regresión del ángulo de rotación directamente en su cabezal de predicción, eliminando la necesidad de procesos de post-procesamiento complejos o redes secundarias.

| Componente | Especificación Técnica | Función en el Proyecto |
| :--- | :--- | :--- |
| **Backbone** | CSPDarknet con bloques C2f | Extracción de características multiescala con flujos de gradiente optimizados para detectar objetos pequeños (motocicletas) y grandes (articulados). |
| **Neck** | Path Aggregation Network (PANet) | Mejora la propagación de información de baja frecuencia (ubicación espacial) y alta frecuencia (semántica de clase) en el flujo de la red. |
| **OBB Head** | Decoupled Head con Regresión Angular | Separa la tarea de clasificación de la de localización. Utiliza una rama específica para predecir el ángulo $\theta$, crucial para el cumplimiento de la métrica rIoU. |
| **Loss Function** | ProbIoU / Complete IoU (CIoU) | Funciones de pérdida diseñadas para manejar la rotación, minimizando la discrepancia entre el ángulo predicho y el real de forma diferenciable. |

## 2.2. Capacidad de Escalamiento para Grandes Volúmenes de Datos

Con un dataset de **50 GB** (aproximadamente cientos de miles de instancias), se requiere una capacidad de parámetros elevada para evitar el *underfitting*. La variante **YOLOv8l-OBB (Large)** ofrece:
*   **Capacidad de Representación:** Suficiente profundidad para aprender las sutilezas de la flota vehicular peruana (por ejemplo, diferencias visuales entre *combi*, *microbus* y *minibus*).
*   **Robustez ante Variabilidad:** El volumen de datos permite que el modelo aprenda características invariantes bajo diferentes condiciones de iluminación y calidad de video propias de las intersecciones en Perú.

## 2.3. Estrategia de Reutilización de Conocimiento (Transfer Learning)

El modelo base no se inicializa de forma aleatoria, sino utilizando pesos pre-entrenados en el dataset **DOTA (Dataset for Object Detection in Aerial Images)**.

1.  **Conocimiento Geométrico Previo:** A diferencia de ImageNet (que es clasificación general), DOTA contiene millones de instancias de objetos rotados vistos desde perspectivas aéreas. Esto proporciona al modelo una comprensión inicial superior sobre cómo proyectar y delimitar objetos con ángulos arbitrarios.
2.  **Adaptación de Dominio:** Los filtros aprendidos para detectar vehículos en DOTA son altamente transferibles al contexto de cámaras de tráfico en intersecciones urbanas, lo que acelera significativamente la convergencia del entrenamiento a pesar del gran tamaño del dataset.
