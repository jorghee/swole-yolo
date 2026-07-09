Para escribir un artículo científico con la estructura exacta que exigen conferencias como SIMBIG (que usualmente siguen el formato Springer LNCS), deben organizar el contenido de inicio a fin en **6 secciones clave**.

Tomando como referencia la excelente organización metodológica del paper `main_es.pdf`, aquí tienen la plantilla exacta y el "esqueleto" de su paper para que comiencen a rellenarlo:

---

## 📄 Estructura Completa del Paper (De Inicio a Fin)

### 📌 Título, Resumen y Keywords

* **Título:** *An Efficient and Lightweight Approach for Oriented Vehicle Detection in Peruvian Urban Intersections* (u otra de las opciones que vimos).
* **Abstract (Resumen):** Un solo párrafo de máximo 150-200 palabras que resume: El problema (tráfico/recursos limitados), la propuesta (vuestro modelo ligero + dataset MTC), la metodología (prueba en PC, Pi 4 y Pi Zero con cuantización) y el resultado principal (ej. *"nuestro modelo alcanzó X FPS en la Pi Zero con solo una pérdida del X% en mAP"*).
* **Keywords:** *Oriented Object Detection, Edge AI, Raspberry Pi, Intelligent Transportation Systems, Model Quantization.*

---

### 1. Introducción

Aquí se vende el problema y se justifica la investigación. Al igual que en `main_es.pdf`, debe cerrarse listando explícitamente sus contribuciones.

* **El Contexto:** Importancia de monitorear el tráfico en ciudades en desarrollo (Perú) usando sistemas de transporte inteligentes (ITS).
* **La Brecha:** Los modelos actuales (como YOLO normal) requieren GPUs caras; las municipalidades necesitan soluciones baratas (Edge AI) instaladas en las calles.
* **El Reto Técnico:** Explicar por qué las cajas horizontales fallan en intersecciones y por qué la detección orientada (OBB) es necesaria.
* **Contribuciones Principales:** Listar en viñetas qué aporta su trabajo:


1. Un pipeline de detección vehicular optimizado para la realidad del parque automotor peruano usando datos del MTC.
2. El diseño de una arquitectura ligera (`TinyOrientedDetector`) optimizada para CPU.
3. Un benchmark multiplataforma exhaustivo que evalúa el rendimiento en PC, Raspberry Pi 4 y Raspberry Pi Zero 2 W.
4. La evaluación del impacto de la cuantización INT8 en la precisión y la latencia en hardware de memoria ultra-limitada (512 MB).



---

### 2. Trabajos Relacionados (Related Work)

Revisión de lo que otros científicos han hecho. Para ganar rigurosidad, sigan la estrategia de `main_es.pdf` y resuman los antecedentes en una tabla comparativa indicando qué limitaciones tenían y cómo se conectan con su propuesta.

* **Sección A:** Trabajos sobre detección de vehículos en tráfico (mencionar algoritmos clásicos y por qué consumen muchos recursos).
* **Sección B:** Detección de objetos orientados (OBB) y su ventaja en curvas o cruces.
* **Sección C:** Despliegue de modelos de Deep Learning en dispositivos embebidos (Raspberry Pi) usando optimización.

---

### 3. Materiales y Métodos (La propuesta técnica)

Esta es la sección de ingeniería donde describen las herramientas y el diseño del experimento.

```
Dataset MTC -> Preprocesamiento (OBB) -> Entrenamiento (PC) -> Cuantización (INT8) -> Despliegue (Raspberry Pi)

```

* **3.1 El Dataset del MTC:** Explicar el origen de los datos, las 9 categorías de vehículos y cómo prepararon los tamaños (300, 3000, 5000 imágenes).


* **3.2 Arquitectura del Modelo Propuesto:** Describir matemáticamente o mediante un diagrama de bloques las capas de su `TinyOrientedDetector`.
* **3.3 Modelo de Referencia (Baseline):** Describir brevemente el modelo YOLO Nano usado para la comparación.
* **3.4 Estrategia de Optimización:** Explicar el proceso de cuantización post-entrenamiento a enteros de 8 bits (INT8) para reducir el uso de memoria (VRAM/RAM).

---

### 4. Experimentos y Resultados (El núcleo numérico)

Aquí colocan las gráficas y las tablas con los datos reales que midieron. Es la parte más importante del artículo.

* **4.1 Configuración del Entorno (Setup):** Describir las especificaciones de la PC, de la Raspberry Pi 4 y de la Raspberry Pi Zero 2 W (mencionando procesadores y limitación de RAM).
* **4.2 Impacto del Tamaño del Dataset:** Mostrar una gráfica (curva de aprendizaje) de cómo el mAP subió a medida que pasaron de 300 a 3000 y 5000 imágenes de entrenamiento.
* **4.3 Resultados del Benchmark de Inferencia:** Colocar la gran tabla comparativa (la que diseñamos en la respuesta anterior) que cruza Dispositivos vs. Modelos vs. FPS, mAP y uso de RAM.

---

### 5. Discusión

Aquí interpretan los números de la sección anterior. Es clave ser honestos y analíticos, una práctica fundamental observada en el análisis de benchmarks compactos.

* Analizar el *Trade-off* (intercambio): ¿Cuánto mAP sacrificamos al cuantizar a INT8 a cambio de ganar FPS en la Raspberry Pi Zero?
* Explicar por qué su modelo superó (o compitió dignamente) contra YOLO Nano en la Raspberry Pi Zero debido a las restricciones de memoria.
* **Limitaciones del estudio:** Admitir las debilidades actuales de forma académica (por ejemplo, que el dataset actual solo incluye tomas diurnas o que falta evaluar el consumo energético en miliamperios de la Raspberry).



---

### 6. Conclusiones y Trabajo Futuro

* **Conclusiones:** Un breve resumen de los hallazgos (ej. *"Se demostró la viabilidad de realizar detección orientada en hardware de $20..."*).
* **Trabajo Futuro:** Plantear los siguientes pasos lógicos (ej. *"Como trabajo futuro, planeamos integrar este pipeline en un nodo de cámara físico en una intersección real de Lima y evaluar modelos secuenciales para el conteo de tráfico en tiempo real"*).



---

### 🤝 Agradecimientos y Referencias

* **Agradecimientos:** Dar crédito al Ministerio de Transportes y Comunicaciones (MTC) de Perú por facilitar el dataset institucional.


* **Referencias:** Formatear la bibliografía en estilo Springer LNCS o IEEE (mínimo unos 15 a 20 artículos científicos indexados, priorizando papers de los últimos 3 años).
