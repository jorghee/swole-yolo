🎯 1. Propósito (El "Por qué" del proyecto)
El objetivo es crear un sistema de Inteligencia Artificial que detecte y clasifique 9 tipos de vehículos en intersecciones urbanas de Perú, usando cajas de detección rotadas (Oriented Bounding Boxes).
La meta final es que este sistema sea tan ligero y eficiente que pueda correr en tiempo real en una Raspberry Pi Zero 2 W (un dispositivo de $20 con solo 512 MB de RAM), ideal para cámaras de tráfico inteligentes de bajo costo.

🛠️ 2. Lo que implementaremos (La estrategia 80/20)
Para no duplicar trabajo innecesariamente, nos enfocaremos en:

El Dataset (MTC Perú): Haremos pruebas con 300, 3,000 y 5,000 imágenes para demostrar científicamente cómo mejora el modelo al tener más datos.

Nuestro Modelo (TinyOrientedDetector): Lo programaremos en un solo framework (el que mejor dominemos, PyTorch o TensorFlow). No hace falta hacer los dos desde cero.

El Modelo de Competencia (Baseline): Usaremos un modelo estándar del estado del arte, como YOLO Nano (v8 o v11), para comparar si nuestro modelo es mejor o peor.

La Optimización Secreta (Cuantización a INT8): Convertiremos ambos modelos a formatos ligeros (como TFLite o ONNX) y reduciremos el peso de sus números a enteros (INT8). Esto toma 3 líneas de código y es lo que permitirá que corran en la Raspberry Pi sin congelarla.

📊 3. Las métricas que mediremos (Los experimentos)
Evaluaremos los modelos en dos campos de batalla: una PC normal y la Raspberry Pi, midiendo lo siguiente:

Métricas de Inteligencia (Precisión)
mAP (Mean Average Precision): Qué tan exactas son las cajas que dibuja el modelo.

Matriz de Confusión: Para ver si el modelo confunde las combis con los autos o las mototaxis.

Métricas de Eficiencia (Recursos)
Tamaño del archivo (MB): Cuánto espacio ocupa el modelo en el disco.

Velocidad de procesamiento (FPS): Cuántos fotogramas por segundo procesa en la PC vs. en la Raspberry Pi.

Uso de Memoria RAM: Cuántos megabytes consume mientras está ejecutándose en la Raspberry.

📝 4. Cosas importantes para el Paper (Cómo asegurar que SIMBIG nos apruebe)
Para que los revisores de la conferencia nos den luz verde, el artículo debe destacar estos 4 pilares:

Detección Orientada (OBB): No usamos rectángulos aburridos; detectamos la inclinación de los carros en las curvas, lo cual es más avanzado.

Realidad Local Peruana: El dataset del MTC refleja el caos vehicular real de nuestro país (combis, mototaxis), algo que los datasets gringos o europeos no tienen.

Hardware Real Extremo: No nos quedamos en simulaciones teóricas; demostramos que corre en una Raspberry Pi Zero real.

Curva de Datos: Demostraremos con gráficas el impacto de pasar de 300 a 5,000 imágenes en el entrenamiento.



---

¡**No dejes de probar en la PC!** Y el hecho de que tengas una **Raspberry Pi 4** es una mina de oro para el paper.

Aquí te aclaro exactamente cómo organizar los dispositivos y los modelos para que el experimento sea perfecto y requiera el menor esfuerzo posible.

---

### 1. ¿Por qué SÍ debes probar en tu PC? (Y cómo usar la Pi 4)

El corazón de un buen paper de Machine Learning es la **comparación**. Si solo muestras resultados en la Raspberry Pi Zero, los revisores no sabrán si el modelo es lento por culpa de la Raspberry o porque el modelo está mal diseñado.

Al probar en la PC, en la Pi 4 y en la Pi Zero, creas un **espectro de hardware** (Alto, Medio y Bajo rendimiento). Esto le da un nivel científico brutal al artículo con muy poco esfuerzo extra (ya que el código para medir la velocidad es el mismo, solo lo copias y pegas en los tres dispositivos).

El rol de cada dispositivo será:

* **Tu PC (Gama Alta / Control):** Sirve para saber cuál es el "máximo potencial" de los modelos en precisión y velocidad sin restricciones.
* **Raspberry Pi 4 (Gama Media / Edge Estándar):** Es el dispositivo Edge más común en el mercado. Veremos cómo se comporta ahí.
* **Raspberry Pi Zero 2 W (Gama Ultra-Baja / El reto extremo):** Aquí es donde demuestran que su modelo destaca, porque probablemente YOLO sufra mucho en este dispositivo, pero el de ustedes seguirá vivo.

---

### 2. ¿En total son solo 2 modelos?

**Sí, a nivel de código y diseño solo trabajarán con 2 arquitecturas:** la suya (`TinyOrientedDetector`) y la de competencia (`YOLO Nano`). No necesitan programar nada más.

Sin embargo, en el paper se presentará como **4 variantes**, porque evaluarás cada modelo en su estado "normal" y en su estado "optimizado" (cuantizado).

Míralo de esta manera:

1. **YOLO Nano (Normal - FP32):** El gigante comercial.
2. **YOLO Nano (Optimizado - INT8):** El gigante intentando ponerse a dieta.
3. **Su Modelo (Normal - FP32):** Su propuesta original.
4. **Su Modelo (Optimizado - INT8):** Su propuesta optimizada para volar en hardware chico.

---

### 📋 La Tabla Definitiva de Experimentos (El mapa de tu Paper)

Para que tu grupo lo vea claro, la sección de resultados del paper consistirá en llenar esta única tabla. Correr los scripts para llenarla les tomará una tarde, pero visualmente es lo que venderá el artículo:

| Dispositivo | Modelo / Arquitectura | Optimización | mAP (¿Detecta bien?) | FPS (¿Va rápido?) | RAM (¿Consume mucho?) |
| --- | --- | --- | --- | --- | --- |
| **PC (CPU)** | 1. YOLO Nano | Ninguna (FP32) | *Máxima* | *Muy rápido* | *No importa* |
|  | 2. **Nuestro Modelo** | Ninguna (FP32) | *Buena* | *Rapidísimo* | *No importa* |
| **Raspberry Pi 4** | 1. YOLO Nano | Cuantizado (INT8) | *Alta* | *Decente* | *Moderado* |
| *(Gama Media)* | 2. **Nuestro Modelo** | Cuantizado (INT8) | *Buena* | **Mejor que YOLO** | **Bajo** |
| **Raspberry Pi Zero 2W** | 1. YOLO Nano | Cuantizado (INT8) | *Baja / Regular* | *Muy lento (¿1-2 FPS?)* | *Casi al límite (512MB)* |
| *(Gama Extrema)* | 2. **Nuestro Modelo** | Cuantizado (INT8) | *Estable* | **¡El ganador! (Ej. 10 FPS)** | **Súper ligero** |

### Conclusión para el grupo:

Hacer las pruebas en la PC y en la Pi 4 no les quitará tiempo (es usar el mismo código de medición que ya hicieron para la Pi Zero). Al contrario, les dará los datos necesarios para justificar por qué su modelo es una excelente solución para la escasez de recursos. ¡Tienen el éxito asegurado con este ecosistema de pruebas!
