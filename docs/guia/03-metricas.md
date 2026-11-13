# 3. Métricas de evaluación

**Objetivo:** entender exactamente qué significan precisión, recall, AP, mAP@50 y mAP@50–95, cómo leer una matriz de confusión y por qué una cifra como "93.9 %" necesita contexto para ser creíble.

## 3.1 TP, FP, FN: el vocabulario base

Elegimos un umbral de IoU (p. ej. 0.50) y emparejamos predicciones con cajas reales (cada real solo puede emparejarse una vez):

| Término | Definición | Ejemplo |
| --- | --- | --- |
| **TP** (verdadero positivo) | predicción de la clase correcta con IoU ≥ umbral con una real aún libre | el modelo marca un auto y hay un auto ahí |
| **FP** (falso positivo) | predicción que no empareja con ninguna real (clase errónea, mal ubicada, o duplicada) | marca un auto sobre un árbol |
| **FN** (falso negativo) | caja real sin ninguna predicción que la empareje | un auto que el modelo no vio |
| *TN* | no se usa en detección: hay infinitas "no-cajas" correctas | — |

\[
\text{Precisión}=\frac{TP}{TP+FP}\qquad\qquad \text{Recall}=\frac{TP}{TP+FN}
\]

- **Precisión**: de lo que el modelo dijo, ¿qué fracción era cierta? (castiga falsas alarmas).
- **Recall** (sensibilidad): de lo que había, ¿qué fracción encontró? (castiga omisiones).

!!! example "Ejemplo numérico"
    En una imagen hay 10 autos. El modelo da 12 cajas: 8 emparejan con autos reales, 4 no.
    TP = 8, FP = 4, FN = 2. Precisión = 8/12 = 0.67; recall = 8/10 = 0.80.

Para contar un solo número que combine ambas se usa **F1** \(=\tfrac{2PR}{P+R}\), la media armónica.

## 3.2 El umbral de confianza cambia todo

El modelo asigna a cada caja una confianza. Si solo aceptamos cajas con confianza > 0.9 tendremos **pocos FP** (precisión alta) pero **muchos FN** (recall bajo); con umbral 0.1 ocurre lo contrario. Ni precisión ni recall por separado describen al modelo: dependen de un umbral arbitrario.

## 3.3 Curva precisión–recall y AP

Si barremos todos los umbrales de confianza obtenemos una curva P(R). El **AP** (*Average Precision*) es el **área bajo esa curva** para una clase. Resume el comportamiento en *todos* los umbrales en un solo número entre 0 y 1.

!!! example "Ejemplo numérico paso a paso (una clase, 4 objetos reales)"
    Ordenamos las predicciones por confianza decreciente y las marcamos:

    | # | ¿TP? | TP acum. | FP acum. | Precisión | Recall |
    | --- | --- | --- | --- | --- | --- |
    | 1 | sí | 1 | 0 | 1.00 | 0.25 |
    | 2 | sí | 2 | 0 | 1.00 | 0.50 |
    | 3 | no | 2 | 1 | 0.67 | 0.50 |
    | 4 | sí | 3 | 1 | 0.75 | 0.75 |
    | 5 | no | 3 | 2 | 0.60 | 0.75 |

    El 4.º objeto real nunca se detecta: el recall máximo es 0.75.
    Se reemplaza la precisión en cada punto por la **máxima precisión a la derecha** (envolvente) y se integra:

    - Recall 0→0.25: precisión 1.00 → área \(0.25\)
    - Recall 0.25→0.50: 1.00 → \(0.25\)
    - Recall 0.50→0.75: 0.75 → \(0.1875\)
    - Recall > 0.75: 0

    **AP = 0.25 + 0.25 + 0.1875 = 0.6875.** (Ultralytics usa una interpolación de 101 puntos de recall estilo COCO; el principio es el mismo.)

### mAP

**mAP** (*mean* AP) es el promedio del AP sobre las clases:

\[
\text{mAP}=\frac{1}{C}\sum_{c=1}^{C}\text{AP}_c
\]

!!! warning "Trampa común: mAP es un promedio *simple* entre clases"
    Cada clase pesa lo mismo, tenga 2 instancias o 3 515. En nuestro paper (Tabla 5) el mAP@50 de 0.9394 es **exactamente el promedio de los 9 AP por clase**: \((0.993+0.924+0.995+0.981+0.749+0.962+0.941+0.995+0.915)/9=0.9394\).
    La clase `articulated_bus` tiene **2 instancias de test** y aporta 1/9 (11 %) al mAP. Si su AP real fuera 0.5 en lugar de 0.995, el mAP@50 bajaría a **0.884**; si fuera 0, a **0.829**. Una sola clase con dos ejemplos mueve el mAP hasta 11 puntos.

## 3.4 mAP@50 vs. mAP@50–95

El AP depende del umbral de IoU que define "correcto":

- **mAP@50** (o AP50): umbral IoU = 0.50. Permisivo: basta con que la caja cubra la mitad.
- **mAP@50–95**: promedio del mAP en 10 umbrales, 0.50, 0.55, …, 0.95 (convención COCO). Premia la **precisión de localización**: a IoU 0.95 la caja debe ser casi perfecta.

| | mAP@50 | mAP@50–95 |
| --- | --- | --- |
| Responde | ¿lo encontró (más o menos bien)? | ¿lo localizó con precisión? |
| Nuestro mejor modelo | 0.939 | 0.817 |

La **diferencia** entre ambos (0.939 − 0.817 = 12 puntos) dice cuánto se pierde al exigir cajas más exactas. Con objetos de ≈ 10 px en la entrada del modelo, exigir IoU 0.9 equivale a acertar el borde con décimas de píxel: por eso el mAP@50–95 es el número más duro y más informativo del paper.

!!! tip "IoU en cajas orientadas"
    Calcular el IoU entre rectángulos *girados* es geométricamente más complejo. Ultralytics lo aproxima con una distancia entre distribuciones gaussianas (**ProbIoU**, [capítulo 4](04-cajas-orientadas.md)). Esto significa que nuestros mAP **no son numéricamente comparables** con los de detectores horizontales ni con benchmarks que usen IoU poligonal exacto.

## 3.5 Precisión y recall que reporta Ultralytics

Los valores `precision` y `recall` de las tablas del paper son resúmenes en **un** umbral de confianza. Según la implementación de Ultralytics (versión usada: 8.4.211), ese umbral es el que **maximiza F1** promediado entre clases; es un detalle que conviene declarar en el paper y verificar en el código antes de afirmarlo.

## 3.6 Matriz de confusión

Tabla donde las columnas son la clase **real** y las filas la clase **predicha** (convención de Ultralytics), más una fila/columna extra `background`:

- Diagonal: aciertos de clase.
- Fila `background`: objetos reales que no se detectaron (FN).
- Columna `background`: predicciones sin objeto real (FP).
- Fuera de la diagonal: confusiones entre clases (p. ej. `minibus` predicho como `van`).

Normalizada por columna, cada columna suma 1 y la lectura es "de los objetos reales de esta clase, ¿a dónde fue a parar cada fracción?".

!!! tip "En nuestro proyecto"
    En la Fig. 3 del paper revisado (modelo de 640 px, **split de test**), el 11 % de las `motorcycle` reales se asignan a *background* (no se detectaron) y el 55 % de los falsos positivos (la columna `background`) se predicen como `motorcycle`. (En la versión original, calculada sobre validación, eran 17 % y 45 %: la diferencia entre ambas muestras ilustra la variabilidad del apartado 3.7.) Recuerda el [tamaño de 4 px](01-fundamentos.md#redimensionado-resize-letterbox): es esperable.

## 3.7 Incertidumbre: una cifra sin intervalo es media verdad

Un resultado de test es **una muestra**. Con otro conjunto de test distinto, el número cambiaría. ¿Cuánto?

!!! example "Ejemplo numérico: proporciones con pocos datos"
    Si un modelo detecta 20 de 20 buses, ¿cuál es su recall "verdadero"? El intervalo de Wilson al 95 % para 20/20 es \([0.84,\,1.00]\). Para 2/2 (nuestra `articulated_bus`) es \([0.34,\,1.00]\). Es decir, con 2 ejemplos, "100 %" es compatible con un recall real de solo 34 %.

!!! failure "Evidencia propia de que esto importa"
    En el subconjunto de 2 500 imágenes, el modelo de 50 épocas obtuvo mAP@50 = **0.860** en validación y **0.914** en test. Ambos conjuntos tienen 376 imágenes, el modelo es el mismo, y la diferencia es de **5.4 puntos**. Eso es ruido de muestreo (qué escenas cayeron en cada split). Cualquier diferencia entre configuraciones menor o similar a ~5 puntos medida en *un solo split* no es concluyente. Esto es exactamente lo que critican los revisores del [plan de revisión](../PLAN.md).

Remedios estándar:

1. **Intervalos de confianza por bootstrap**: remuestrear las imágenes de test con reemplazo 1 000 veces y recalcular la métrica; el rango que contiene el 95 % central es el IC.
2. **Varias semillas** (inicializaciones y splits distintos) y reportar media ± desviación.
3. **Test fijo común** para comparar configuraciones.
4. Reportar `n` (instancias) junto a cada AP por clase y marcar las clases con \(n\) pequeño como exploratorias.

## 3.8 Métricas del lado de despliegue

Separadas de la calidad de detección:

| Métrica | Qué mide | Capítulo |
| --- | --- | --- |
| **Latencia** (ms/imagen) | tiempo de una predicción | [7](07-edge.md) |
| **FPS** | \(1000/\text{latencia}_{ms}\) con batch 1 | 7 |
| **P95** | latencia que el 95 % de las imágenes no supera | 7 |
| Tamaño del modelo (MB), parámetros | memoria | 7 |

## Resumen

- Precisión = no inventar; recall = no omitir; AP las integra sobre todos los umbrales de confianza; mAP promedia clases por igual.
- mAP@50 es permisivo; mAP@50–95 exige localización fina.
- Con pocos ejemplos o un solo split, las diferencias pequeñas no son evidencia.

!!! question "Comprueba que entendiste"
    1. Un modelo da 30 detecciones: 24 correctas. Hay 40 objetos reales. ¿Precisión y recall?
    2. ¿Por qué mAP@50–95 ≤ mAP@50 siempre?
    3. En nuestro paper, ¿por qué es engañoso el AP de `articulated_bus` = 0.995?
    4. ¿Qué hay que añadir a una tabla de resultados para que una diferencia de 2 puntos sea interpretable?

    ??? success "Respuestas"
        1. Precisión \(24/30=0.80\); recall \(24/40=0.60\).
        2. Porque el promedio incluye umbrales más estrictos (0.55…0.95) en los que el AP es menor o igual que a 0.50.
        3. Solo hay 2 instancias de test: el valor es muy inestable (IC 95 % del recall de 2/2 ≈ [0.34, 1.00]).
        4. Un intervalo de confianza o desviación entre semillas, y el tamaño de la muestra de test.

## Para profundizar

- Padilla, Netto, da Silva. *A Survey on Performance Metrics for Object-Detection Algorithms*, IWSSIP 2020.
- Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014 (referencia [3] del paper; define mAP@50–95).
- Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993.
