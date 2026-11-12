# 1. Fundamentos de aprendizaje automático y visión

**Objetivo:** entender qué es una imagen para una computadora, qué es una red neuronal convolucional y qué significa "entrenar" un modelo. Sin esto, nada de lo demás tiene sentido.

## 1.1 Una imagen es un arreglo de números

Para una computadora, una fotografía a color es una tabla tridimensional de enteros:

\[
\text{imagen} \in \{0,1,\dots,255\}^{\,H \times W \times 3}
\]

- **H** y **W**: alto y ancho en **píxeles**.
- **3 canales**: rojo, verde y azul (RGB). Cada número dice cuánta luz de ese color hay en ese píxel (0 = nada, 255 = máximo).

!!! example "Ejemplo numérico"
    Nuestras imágenes miden 1920 × 1080 píxeles. Tienen \(1920 \times 1080 \times 3 = 6\,220\,800\) números. Un auto que ocupa 35 × 87 píxeles (la mediana en nuestro dataset) se describe con unos \(35 \times 87 \times 3 \approx 9\,100\) de esos números: **0.15 %** de la imagen.

El modelo nunca "ve" un auto; recibe millones de números y debe aprender qué patrones numéricos corresponden a un auto.

### Redimensionado (resize / letterbox)

Las redes esperan entradas de tamaño fijo y, en la práctica, múltiplo de 32. Antes de entrar al modelo, la imagen se reduce. Ultralytics conserva la proporción: escala el **lado largo** a `imgsz` y rellena con gris si hace falta (*letterbox*).

!!! example "Ejemplo numérico: cuánto se encoge todo"
    - A `imgsz = 544`: factor \(544/1920 = 0.283\). Un auto de 35 px de lado corto queda en **≈ 9.9 px**; una moto (12.8 px) en **≈ 3.6 px**.
    - A `imgsz = 640`: factor \(0.333\). El auto queda en 11.7 px y la moto en 4.3 px.

    Una moto de 4 píxeles es casi un punto. Esto explica buena parte de por qué `motorcycle` es la clase más difícil del paper (AP@50 = 0.749). Volveremos a esto en el [capítulo 3](03-metricas.md) y en el [7](07-edge.md).

!!! warning "Trampa común"
    `imgsz = 540` no es un tamaño válido: no es múltiplo de 32. Ultralytics lo corrige a **544** y avisa (`imgsz=[540] must be multiple of max stride 32, updating to [544]`). Por eso el paper habla de "540 solicitado, 544 efectivo".

## 1.2 Aprendizaje supervisado

En **aprendizaje supervisado** le damos al modelo ejemplos de **entrada → respuesta correcta** y le pedimos que aprenda la regla.

| Elemento | En nuestro proyecto |
| --- | --- |
| Entrada \(x\) | una imagen de 1920 × 1080 |
| Respuesta correcta \(y\) (*etiqueta*, *ground truth*, *GT*) | la lista de vehículos: clase + caja orientada |
| Modelo \(f_\theta\) | YOLO11n-OBB |
| Parámetros \(\theta\) | 2 655 478 números que el entrenamiento ajusta |
| Predicción \(\hat y = f_\theta(x)\) | las cajas y clases que el modelo cree ver |

Un **modelo** es simplemente una función con muchos botones (parámetros). **Entrenar** es girar esos botones hasta que las predicciones se parezcan a las etiquetas.

## 1.3 La neurona y la red

Una **neurona** calcula una suma ponderada y la pasa por una función no lineal:

\[
a = \sigma\!\left(\sum_i w_i x_i + b\right)
\]

- \(w_i\) y \(b\) son parámetros (pesos y sesgo).
- \(\sigma\) es la **función de activación**. Sin ella, apilar capas equivaldría a una sola operación lineal y la red no podría aprender nada complejo. YOLO11 usa **SiLU**: \(\sigma(x)=x\cdot\text{sigmoid}(x)\).

Una **red neuronal** apila muchas capas de neuronas: la salida de una capa es la entrada de la siguiente. "Profundo" (*deep*) significa muchas capas.

## 1.4 La convolución: la idea clave para imágenes

Conectar cada píxel con cada neurona sería inmanejable (6.2 millones de entradas). La **convolución** aprovecha dos hechos: los patrones útiles son **locales** (un borde ocupa pocos píxeles) y **se repiten** por toda la imagen.

Un **kernel** (filtro) es una mini-ventana de pesos, por ejemplo 3 × 3. Se desliza por la imagen y en cada posición calcula una suma ponderada:

!!! example "Ejemplo numérico: un detector de bordes verticales"
    Kernel \(K=\begin{pmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{pmatrix}\) sobre un parche con un borde (izquierda oscura = 10, derecha clara = 90):

    \[
    \begin{pmatrix}10&10&90\\10&10&90\\10&10&90\end{pmatrix}
    \;\Rightarrow\;
    (-10+90)+(-10+90)+(-10+90)=240
    \]

    En una zona uniforme (todo 50) el resultado sería 0. El kernel "se enciende" donde hay un borde vertical. En una red real **los valores del kernel no los escribe una persona: los aprende el entrenamiento**.

Parámetros de una convolución:

| Parámetro | Significado |
| --- | --- |
| **kernel size** \(k\) | tamaño de la ventana (3 en casi todo YOLO) |
| **stride** \(s\) | cuántos píxeles salta la ventana. \(s=2\) reduce a la mitad el tamaño |
| **padding** \(p\) | borde de ceros añadido para controlar el tamaño de salida |
| **canales de entrada / salida** | cuántos mapas entran y cuántos filtros distintos se aprenden |

Tamaño de salida:

\[
\text{out} = \left\lfloor \frac{\text{in} + 2p - k}{s} \right\rfloor + 1
\]

Número de parámetros de una convolución (con *batch normalization* en lugar de sesgo):

\[
\text{params} = k \cdot k \cdot C_{\text{in}} \cdot C_{\text{out}} + 2\,C_{\text{out}}
\]

!!! tip "En nuestro proyecto: verifica esta fórmula en nuestro modelo"
    La capa 0 de YOLO11n-OBB es una convolución \(3\to16\) canales, \(k=3\), \(s=2\). Parámetros: \(3\cdot3\cdot3\cdot16 + 2\cdot16 = 432+32 = \mathbf{464}\). Es exactamente el número que reporta `ultralytics` para esa capa. Y con \(s=2\), \(p=1\): una entrada de 640 produce \(\lfloor(640+2-3)/2\rfloor+1=320\).

### Del píxel al concepto: jerarquía de características

Al apilar convoluciones, las primeras capas detectan bordes y colores; las siguientes, texturas y esquinas; las últimas, partes de objetos (una rueda, un techo) y objetos completos. Cada capa "ve" una región mayor de la imagen original: su **campo receptivo**.

### Reducción de resolución y canales

Cada convolución con \(s=2\) divide ancho y alto entre 2 y se compensa **aumentando los canales**. Así la red pasa de "mucha resolución espacial, poca información por píxel" a "poca resolución, mucha información por celda".

!!! tip "En nuestro proyecto"
    YOLO11 produce mapas de características con *stride* 8, 16 y 32 (la imagen reducida 8, 16 y 32 veces). Para `imgsz=640` son cuadrículas de 80×80, 40×40 y 20×20 celdas. Cada celda del mapa de stride 8 "mira" un cuadrado de 8×8 píxeles de la entrada. Una moto de 4 px cabe en menos de una celda: de ahí el problema.

## 1.5 Entrenar: pérdida, gradiente y optimización

1. **Pérdida** (*loss*) \(L(\theta)\): un único número que mide *cuán mal* predice el modelo respecto a las etiquetas. Cero sería perfecto.
2. **Gradiente** \(\nabla_\theta L\): para cada parámetro, indica hacia dónde moverlo para que la pérdida baje. Se calcula con **retropropagación** (*backpropagation*), que aplica la regla de la cadena capa por capa de salida a entrada.
3. **Descenso de gradiente**: actualiza \(\theta \leftarrow \theta - \eta\,\nabla_\theta L\), donde \(\eta\) es la **tasa de aprendizaje** (*learning rate*).
4. Repetir miles de veces.

!!! example "Ejemplo numérico mínimo"
    Un modelo con un solo parámetro: \(\hat y = w\,x\). Dato: \(x=2\), \(y=10\). Pérdida \(L=(wx-y)^2\). Con \(w=1\): \(L=(2-10)^2=64\). Gradiente: \(dL/dw = 2(wx-y)x = 2(-8)(2)=-32\). Con \(\eta=0.1\): \(w \leftarrow 1 - 0.1(-32) = 4.2\). Ahora \(L=(8.4-10)^2=2.56\). Bajó de 64 a 2.56 en un paso.

Vocabulario de entrenamiento (aparece constantemente en el proyecto):

| Término | Definición | Valor en el paper |
| --- | --- | --- |
| **Muestra** | una imagen con sus etiquetas | 1 748 de entrenamiento (subset 2 500) |
| **Batch** | grupo de muestras procesadas juntas para un paso | 32 |
| **Iteración** | un paso de actualización de parámetros | \(\lceil 1748/32 \rceil = 55\) por época |
| **Época** (*epoch*) | una pasada completa por todo el train | 50 ó 70 |
| **Optimizador** | regla de actualización (SGD, Adam, AdamW) | `auto` (lo elige Ultralytics) |
| **Hiperparámetro** | lo que *tú* eliges antes de entrenar (batch, épocas, lr, imgsz) | ver `args.yaml` |
| **Parámetro** | lo que el entrenamiento ajusta | 2 655 478 |

!!! tip "En nuestro proyecto"
    70 épocas × 55 iteraciones = **3 850 pasos de actualización**. Es muy poco comparado con entrenar desde cero (cientos de miles de pasos). Funciona porque partimos de un modelo ya entrenado: ver [transfer learning](05-yolo11.md#52-transfer-learning).

## 1.6 Generalizar: sobreajuste y los tres conjuntos

El objetivo no es acertar en las imágenes de entrenamiento (que el modelo ya "vio"), sino en imágenes **nuevas**. Un modelo que memoriza el entrenamiento y falla en lo nuevo sufre **sobreajuste** (*overfitting*).

Para medirlo se parten los datos en tres conjuntos **disjuntos**:

| Conjunto | Para qué sirve | Se usa… |
| --- | --- | --- |
| **Train** | ajustar los parámetros | en cada iteración |
| **Validación** (*val*) | vigilar el progreso y **elegir** (mejor época, hiperparámetros) | en cada época |
| **Test** | medir el rendimiento final, honesto | **una sola vez** |

!!! warning "Trampa común"
    Si usas el test para tomar decisiones (elegir época, ajustar umbrales, escoger entre modelos) deja de ser un examen limpio: te convertiste en parte del entrenamiento. Nuestro código evalúa en test una vez con el mejor checkpoint (`finetune.py`). Pero el paper muestra las Figs. 2 y 3 con el split de **validación**, que sí se usó para elegir el checkpoint: otra razón por la que hay que corregirlo (ver [plan](../PLAN.md)).

La **detención temprana** (*early stopping*, `patience=20`) corta el entrenamiento si la validación no mejora en 20 épocas seguidas. En nuestros runs nunca se activó: el mejor checkpoint cayó cerca del final (épocas 43–50 de 50), señal de que **el modelo seguía mejorando** y probablemente no había convergido.

## 1.7 Hardware: CPU, GPU y por qué importa

- **GPU**: miles de núcleos simples; ideal para las multiplicaciones de matrices del entrenamiento. El entrenamiento se hizo en un VPS (`run_finetune.sh` está pensado para uno); los logs no conservan el modelo de GPU, aunque los tiempos (≈ 9 min para 50 épocas) son compatibles con GPU.
- **CPU**: pocos núcleos potentes. La **Raspberry Pi 4** solo tiene CPU (4 núcleos ARM Cortex-A72). No hay GPU utilizable.
- **FLOPs**: operaciones de punto flotante necesarias para una predicción. Es una medida de coste computacional *independiente del hardware*. Ultralytics imprime **6.6 GFLOPs** para YOLO11n-OBB al cargarlo (la resolución de referencia de esa cifra no está documentada en nuestros logs). Lo que sí es seguro: los FLOPs crecen con el **número de píxeles**, es decir, con el cuadrado de `imgsz`.

## Resumen

- Imagen = tensor de números; el modelo es una función con millones de parámetros.
- Convolución = filtro local aprendido; apilarlas construye conceptos complejos.
- Entrenar = reducir una pérdida con descenso de gradiente.
- Siempre medir en datos que el modelo no usó para decidir nada (test).

!!! question "Comprueba que entendiste"
    1. ¿Cuántos parámetros tiene una convolución 3×3 de 32 a 64 canales (con BN)?
    2. Si un auto mide 35 px de lado corto en la imagen original, ¿cuánto mide tras reducir a `imgsz=640`?
    3. ¿Por qué no es válido elegir el mejor modelo mirando el test?
    4. ¿Qué indica que el mejor checkpoint sea de la época 50 de 50?

    ??? success "Respuestas"
        1. \(3\cdot3\cdot32\cdot64 + 2\cdot64 = 18\,432+128 = 18\,560\).
        2. \(35 \times 640/1920 \approx 11.7\) px.
        3. Porque la elección usa información del test; la medida final queda optimista y ya no estima el rendimiento en datos realmente nuevos.
        4. Que el modelo aún mejoraba al detenerse; probablemente faltaban épocas y la comparación con una configuración de 70 épocas mezcla "más épocas" con otros cambios.

## Para profundizar

- CS231n, módulos 1–2 (clasificación lineal, optimización, convoluciones).
- *Dive into Deep Learning*, capítulos 7 (CNN modernas) y 3–4 (fundamentos).
