# 3. Preparación y Gestión del Dataset

Con un volumen de 50 GB aproximadamente y una métrica de evaluación basada en **Macro AP**, la gestión de datos es la fase con mayor impacto en el rendimiento final.

## 3.1. Ingesta y Extracción de Datos (Video-to-Frames)
Dado que el input son clips de 5s a 10 FPS, cada clip genera 50 imágenes. Para evitar el sobreajuste por redundancia (frames casi idénticos), implementamos:
*   **Muestreo Estratégico:** Extracción de frames clave. En producción, se recomienda un *stride* de muestreo dinámico basado en el movimiento detectado para asegurar que cada imagen aporte información nueva al gradiente.

## 3.2. Estructuración del Formato OBB
El formato de entrada proporcionado ($cx, cy, w, h, \text{angle\_deg}$) debe mapearse al formato esperado por el motor de entrenamiento (formato de 4 puntos o formato normalizado).

**Conversión Técnica:**

$$(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4) \leftarrow f(cx, cy, w, h, \theta)$$

Es crítico asegurar que el ángulo se procese en el rango correcto ($[0, \pi]$ o $[-\pi/2, \pi/2]$) según la implementación específica de la función de pérdida del modelo.

## 3.3. Estrategia de Balanceo de Clases (Macro AP Focus)
La métrica **Macro AP** penaliza severamente el mal desempeño en clases raras. Con 9 clases (desde "auto" hasta "articulado"), aplicaremos:
*   **Re-sampling:** Duplicar la frecuencia de aparición de clips que contengan clases minoritarias (*mototaxi*, *articulado*, *combi*) en el set de entrenamiento.
*   **Filtrado por Dificultad:** Utilizar técnicas de *Hard Example Mining* para identificar frames donde el modelo actual falla (objetos pequeños o muy rotados) y aumentar su peso en el dataset de entrenamiento.

## 3.4. División del Dataset (Splitting)
La separación se realiza a **nivel de clip**, no de frame.
*   **Train (80%):** Diversidad de intersecciones y condiciones climáticas.
*   **Validation (20%):** Debe incluir al menos dos intersecciones completas que el modelo no haya visto en *Train* para validar la capacidad de generalización a nuevas infraestructuras viales.
