## 2. Ingeniería de Datos y Construcción de `PyDataset`

Una vez seleccionados los identificadores de los frames representativos (0, 25, 49), el siguiente desafío técnico es la ingesta eficiente de estos datos. En Keras 3, la gestión de datos debe abstraerse de la memoria RAM para garantizar que el sistema no colapse al escalar de la muestra de 300 imágenes al dataset de 50 GB.

### A. Decisión Técnica: Implementación de `keras.utils.PyDataset`
Para este proyecto, se ha optado por construir una clase personalizada que herede de `keras.utils.PyDataset` en lugar de utilizar arreglos simples de NumPy o generadores básicos de Python.

*   **Motivo de la elección:** `PyDataset` es la interfaz de datos recomendada en Keras 3 para garantizar el soporte multi-backend (TensorFlow, PyTorch, JAX). A diferencia de los generadores estándar, `PyDataset` es seguro para el subprocesamiento múltiple (*multiprocessing*), lo que permite que la CPU preprocese el siguiente lote (*batch*) mientras la GPU procesa el actual.
*   **Ventajas:**
    *   **Lazy Loading (Carga Perezosa):** Las imágenes permanecen en el disco y solo se cargan en la RAM en el momento exacto en que se necesitan para un paso de entrenamiento.
    *   **Acceso por Índice:** Permite barajar (*shuffling*) los datos de forma eficiente en cada época, lo cual es vital para la convergencia del gradiente.
    *   **Estandarización:** Facilita la aplicación de las mismas transformaciones tanto al set de entrenamiento como al de validación y test.
*   **Limitaciones:** Requiere una implementación inicial más rigurosa que un simple `ImageDataGenerator`, ya que debemos gestionar manualmente el mapeo entre los IDs del CSV y las rutas físicas de los archivos.

### B. Estandarización de Tensores (Resize y Normalización)
Dentro del método `__getitem__` de nuestra clase, se aplican transformaciones críticas "al vuelo" (*on-the-fly*):

1.  **Redimensionamiento (Target Size):** Se establece una resolución estándar de **224x224 píxeles**.
    *   *Justificación:* Es el equilibrio óptimo para una CNN construida desde cero. Resoluciones menores (por ejemplo, 64x64) pierden detalles clave de las clases pequeñas (motocicletas), mientras que resoluciones mayores (por ejemplo, 512x512) aumentarían exponencialmente el número de parámetros y el tiempo de entrenamiento sin un beneficio claro en una arquitectura inicial.
2.  **Normalización de Intensidad:** Los valores de los píxeles $[0, 255]$ se escalan al rango **$[0, 1]$**.
    *   *Impacto:* Esto asegura que los pesos de la red no exploten durante el *backpropagation* debido a magnitudes de entrada elevadas, facilitando una convergencia más rápida y estable.
3.  **Codificación de Etiquetas (Label Encoding):** Conversión de las clases del CSV (1-9) a un formato compatible con la función de pérdida (usualmente índices 0-8).

### C. Configuración de Lotes (Batching) y Shuffling
Para la muestra de 300 imágenes, el impacto es menor, pero la lógica se diseña pensando en los 50 GB:

*   **Batch Size:** Se establece inicialmente en **32**. Es un tamaño estándar que ofrece una buena estimación del gradiente sin exceder la memoria VRAM de GPUs de gama media.
*   **Shuffle:** Se activa únicamente para el conjunto de entrenamiento. En visión artificial, el orden en que se presentan las imágenes puede sesgar el aprendizaje; barajar los datos asegura que el modelo no aprenda secuencias temporales espurias del video, sino características visuales generales.

### D. Consideraciones para la Escalabilidad Masiva
Al trabajar con los 50 GB de datos reales, esta arquitectura de `PyDataset` permite dos optimizaciones de nivel senior:
1.  **Workers y Multiprocessing:** Keras 3 permite definir `workers > 1` en el método `.fit()`. Esto significa que múltiples núcleos de la CPU estarán decodificando imágenes JPG simultáneamente, eliminando el "cuello de botella de I/O" donde la GPU se queda inactiva esperando datos.
2.  **Caché de Memoria Dinámica:** Se puede implementar un sistema de caché para almacenar en RAM las imágenes más frecuentes (si el hardware lo permite), reduciendo el tiempo de lectura de disco en épocas avanzadas.

### E. Impacto en el Pipeline Completo
Este componente de ingeniería de datos garantiza que el resto del pipeline sea agnóstico al tamaño del dataset. La CNN (Punto 3) recibirá tensores de forma $(32, 224, 224, 3)$ independientemente de si estamos entrenando con la muestra de 300 imágenes o con el total de 50 GB, permitiendo una transición fluida y sin cambios en el código de la arquitectura del modelo.
