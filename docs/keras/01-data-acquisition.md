## 1. Adquisición de Datos y Muestreo Estratégico Temporal

La primera fase del pipeline se enfoca en la transición de clips de video a un dataset de imágenes estáticas optimizado. Dado que el dataset original (50 GB) presenta una altísima redundancia temporal (50 frames por cada 5 segundos), procesar cada frame es técnicamente contraproducente: saturaría el almacenamiento y causaría un sobreajuste por similitud visual extrema entre muestras contiguas.

### A. Decisión Técnica: Selección por Intervalos Fijos (0, 25, 49)
Se ha seleccionado un enfoque de muestreo manual reproducible, extrayendo tres frames por clip: el primero (index 0), el central (index 25) y el último (index 49).

*   **Motivo de la elección:** En el análisis de tráfico vehicular a 10 FPS, un intervalo de 2.5 segundos (25 frames) garantiza una varianza espacial suficiente. Los vehículos habrán avanzado distancias considerables, cambiado de escala (perspectiva) y, potencialmente, de ángulo de rotación. Esta estrategia maximiza el aprendizaje de características sin requerir la complejidad computacional de un algoritmo de selección dinámica (como el cálculo de *Optical Flow* o *Inter-frame Difference*).
*   **Ventajas:**
    *   **Determinismo:** El proceso es 100% reproducible en cualquier entorno.
    *   **Bajo costo computacional:** No requiere procesar los píxeles de los 50 frames para decidir cuáles usar; simplemente se accede al índice del archivo.
    *   **Balanceo de contexto:** Captura la entrada, el tránsito por el centro de la intersección y la salida del vehículo.
*   **Limitaciones:** En clips con congestión extrema (tráfico detenido), los tres frames podrían ser visualmente idénticos, aportando información redundante. Sin embargo, para un dataset de 50 GB, el impacto negativo de esto es estadísticamente insignificante comparado con el ahorro en procesamiento.

### B. Implementación de Identificadores (Parsing de IDs)
El sistema debe ser capaz de filtrar los registros del CSV basándose en la estructura del nombre del archivo. Los identificadores siguen el patrón `v_[uuid]_[frame_index]`.

*   **Impacto en el pipeline:** Al filtrar los IDs en el CSV antes de la carga, reducimos el uso de memoria RAM desde el inicio. El dataframe resultante solo contendrá los metadatos necesarios para las imágenes seleccionadas.
*   **Consideración de Escalabilidad:** Para procesar los 50 GB, no se cargarán las imágenes físicamente en esta etapa. Solo se construye un **Índice de Rutas de Archivo** (File Path Index). Esto permite que el sistema de archivos gestione la búsqueda de datos solo cuando el modelo los solicita durante el entrenamiento.

### C. Validación de Integridad y Consistencia
Antes de pasar al entrenamiento, se implementa una auditoría de datos sobre la muestra de 300 imágenes:
1.  **Existencia física:** Verificación de que cada `Id` en el CSV filtrado tenga un archivo `.jpg` correspondiente en el disco.
2.  **Validación de etiquetas:** Filtrado de registros con valor `Target = "none"` para asegurar que la CNN propia aprenda inicialmente sobre clases positivas (si el objetivo es clasificación pura) o manejo de clases de fondo si se desea robustez.
3.  **Detección de corrupción:** Apertura rápida del header de cada imagen para asegurar que el archivo no esté truncado, evitando que el entrenamiento falle inesperadamente en el dataset completo.

### D. Impacto en la Comparativa Multimodelo
Esta etapa de adquisición define el "Gold Standard" del dataset. Al fijar exactamente qué 3 imágenes se usan por cada clip, garantizamos que cuando el modelo se entrene en PyTorch o se evalúe en Scikit-learn, el rendimiento observado sea atribuible exclusivamente a la **arquitectura del modelo** y no a una variación en la calidad o cantidad de los datos recibidos.
