## 4. Estrategia de Optimización y Compilación

Una vez definida la topología de la CNN, el siguiente paso crítico es la configuración del motor de entrenamiento. En Keras 3, la compilación no es solo un paso formal, sino el momento en que se define cómo el modelo aprenderá de los errores y cómo se comportará en el hardware seleccionado (TensorFlow, PyTorch o JAX).

### 4.1. Selección del Backend en Keras 3

Una de las ventajas competitivas de Keras 3 es su agnósticos al backend. Para este proyecto, la elección del backend impacta directamente en la velocidad de compilación y ejecución.

| Backend | Ventaja Principal | Decisión Técnica |
| :--- | :--- | :--- |
| **TensorFlow** | Ecosistema maduro y herramientas de despliegue (TFLite, TFServing). | **Seleccionado por defecto** para compatibilidad con el pipeline de producción y manejo de archivos grandes. |
| **JAX** | Inferencia y entrenamiento extremadamente rápidos gracias a la compilación XLA. | Recomendado para la fase de escala a 50 GB si se dispone de infraestructuras TPU o clusters de GPUs. |
| **PyTorch** | Facilidad de depuración y acceso a un ecosistema de investigación vasto. | Útil si se requiere una comparación directa de bajo nivel con la implementación de PyTorch puro (Punto 11). |

### 4.2. El Optimizador: Algoritmo de Aprendizaje

El optimizador es el encargado de actualizar los pesos de la red basándose en el gradiente de la función de pérdida. Se ha seleccionado **Adam (Adaptive Moment Estimation)** con una tasa de aprendizaje inicial de $1 \times 10^{-3}$.

*   **Justificación:** Adam combina las ventajas de *AdaGrad* (manejo de gradientes dispersos) y *RMSProp* (manejo de objetivos no estacionarios). Es el estándar para arquitecturas CNN desde cero debido a su capacidad de adaptar la tasa de aprendizaje para cada parámetro de forma individual, lo que acelera significativamente la convergencia en las primeras épocas.
*   **Limitaciones:** En etapas muy avanzadas de entrenamiento con datasets masivos (50 GB), Adam puede presentar problemas de generalización frente a **SGD con Momentum**. 
*   **Impacto en el Pipeline:** Se implementará un *Learning Rate Scheduler* en el siguiente punto para mitigar la agresividad de Adam conforme el modelo se acerque al mínimo global.

### 4.3. Función de Pérdida (Loss Function)

Para un problema de clasificación multiclase con 9 categorías vehiculares, la elección estándar es **Categorical Cross-Entropy**.

*   **Matemática del error:** Esta función mide la divergencia entre la distribución de probabilidad predicha por la capa Softmax y la distribución real (etiquetas en formato *one-hot encoding*).
*   **Decisión de Ingeniería:** Si las etiquetas se mantienen como enteros en el CSV para ahorrar memoria, se utilizará **Sparse Categorical Cross-Entropy**.
    *   *Ventaja:* Evita la creación de matrices *one-hot* dispersas en la RAM, algo vital cuando escalamos el volumen de datos, optimizando el uso de memoria durante el cálculo del gradiente.

### 4.4. Métricas de Evaluación en Tiempo Real

Durante la compilación, se definen las métricas que el ingeniero monitoreará para evaluar la salud del entrenamiento.

1.  **Accuracy (Precisión Global):** Proporciona una visión general del rendimiento, pero puede ser engañosa si existe desbalance de clases (por ejemplo, muchas imágenes de "auto" y pocas de "articulado").
2.  **Top-K Accuracy (K=3):** Dado que algunas categorías son visualmente similares, esta métrica evalúa si la clase correcta se encuentra entre las 3 predicciones con mayor probabilidad.
3.  **Precision y Recall (por clase):** Esencial para el análisis comparativo posterior, permitiendo identificar si el modelo es particularmente débil en alguna categoría específica del *SMART Challenge*.

### 4.5. Consideraciones para el Dataset Completo (50 GB)

La compilación en Keras 3 permite integrar la **Compilación XLA (Accelerated Linear Algebra)** mediante el argumento `jit_compile=True` en el método `model.compile()`.

*   **Impacto:** XLA fusiona operaciones de GPU (kernels), reduciendo las escrituras en memoria intermedia. Al escalar a 50 GB, esto puede representar una mejora del **20% al 40% en la velocidad de entrenamiento**, permitiendo procesar más imágenes por segundo y reduciendo los costos de infraestructura en la nube.

### 4.6. Resumen de Compilación

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_acc')],
    jit_compile=True # Optimización para hardware acelerado
)
```

Este paso asegura que el modelo esté listo para recibir los datos del `PyDataset` (Punto 2) y ajustar sus pesos según la arquitectura diseñada (Punto 3), estableciendo las bases para un ciclo de entrenamiento controlado y profesional.
