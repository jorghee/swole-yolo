# 8. Metodología experimental

**Objetivo:** aprender a diseñar experimentos cuyas conclusiones se sostengan, y a leer críticamente un artículo (incluido el nuestro). Este capítulo convierte las críticas de los revisores en principios generales.

## 8.1 Anatomía de un experimento

| Elemento | Pregunta | Ejemplo en nuestro paper |
| --- | --- | --- |
| **Hipótesis** | ¿Qué creo que pasa y por qué? | "más imágenes de dominio mejoran el mAP" |
| **Variable independiente** (lo que cambio) | ¿Qué manipulo? | tamaño del train; resolución; épocas |
| **Variable dependiente** (lo que mido) | ¿Qué observo? | mAP@50, mAP@50–95, latencia |
| **Controles** (lo que mantengo igual) | ¿Qué debe ser idéntico? | modelo, optimizador, test, hardware |
| **Línea base** (*baseline*) | ¿Comparado con qué? | **ninguna** en el paper |
| **Repeticiones** | ¿Cuánto varía por azar? | una sola ejecución por configuración |

La regla de oro: **cambia una sola cosa a la vez**. Si cambias dos, no sabes cuál causó la diferencia.

## 8.2 Confusores: el error más frecuente

Un **confusor** es un factor que cambia junto con la variable de interés y contamina la conclusión. En nuestro paper hay tres.

=== "Escalado de datos"

    **Se quiere medir:** efecto del tamaño del train.
    **Pero también cambia:** el conjunto de test (150 → 452 imágenes, con clips distintos), y la semilla del split.
    **Consecuencia:** las diferencias de la Tabla 3 mezclan *más datos* con *otro examen*. Prueba: el mismo modelo saca 0.860 en val y 0.914 en test dentro de un mismo subconjunto.
    **Arreglo:** un test fijo para todos.

=== "544 px / 50 ép. vs 640 px / 70 ép."

    **Se quiere medir:** efecto de la resolución.
    **Pero también cambia:** el número de épocas (50 → 70).
    **Agravante:** los modelos de 50 épocas tenían su mejor checkpoint en las épocas 43–50: **seguían mejorando**. Parte de la ganancia de +2.5 puntos de mAP@50 puede ser simplemente "20 épocas más".
    **Arreglo:** diseño factorial 2×2 (resolución × épocas), o entrenar todos hasta convergencia.

=== "OBB vs. caja horizontal"

    **Se quiere afirmar:** que las cajas orientadas son ventajosas.
    **Pero no se comparó** con detección horizontal sobre los mismos datos, y el 81 % de las cajas ya están alineadas con los ejes.
    **Arreglo:** entrenar un YOLO11n-HBB equivalente y comparar (idealmente también en el subconjunto de vehículos inclinados).

## 8.3 Líneas base y ablaciones

- **Baseline**: referencia contra la que se mide la mejora. Sin ella "93.9 %" es una cifra aislada: ¿es mucho o poco? Opciones razonables para este trabajo:
    - La misma red **sin preentrenamiento** (desde cero): mide el valor del transfer learning. *(Ya existe un baseline desde cero en `archive/`, pero con otra red; no sirve tal cual.)*
    - **YOLO11n-HBB**, **YOLOv8n-OBB**, **YOLO11s-OBB**: miden valor de OBB y de arquitectura/tamaño.
    - Un baseline "trivial" (p. ej. predecir siempre la clase `car`).
- **Ablación**: quitar o cambiar *un* componente y medir cuánto empeora. Ejemplos aquí: sin mosaic; sin escala; con rotación; sin preentrenamiento.

## 8.4 Azar y estadística básica

El entrenamiento tiene componentes aleatorios (inicialización de la cabeza, orden de los batches, aumentos). Dos ejecuciones idénticas con semillas distintas dan resultados algo distintos.

Procedimiento mínimo:

1. Repetir cada configuración con \(n\ge5\) **semillas**.
2. Reportar **media ± desviación estándar** (o intervalo de confianza al 95 %).
3. Para comparar A vs. B sobre las *mismas* imágenes de test, usar un **bootstrap pareado**: remuestrear imágenes con reemplazo, calcular la diferencia de mAP en cada remuestreo, y mirar si el intervalo incluye 0.
4. Distinguir **significancia** (¿es real?) de **tamaño del efecto** (¿importa?).

!!! example "Ejemplo numérico: ¿cuándo es creíble una diferencia?"
    Supón que con 5 semillas, el mAP@50 de la configuración A es \(0.914\pm0.012\) y el de B \(0.939\pm0.010\). Las medias difieren en 2.5 puntos y la desviación es ≈ 1 punto: la diferencia (≈ 2.5 desviaciones) es probablemente real. Si en cambio la desviación fuera 3 puntos, 2.5 puntos serían indistinguibles del azar. **Sin repetir no hay forma de saber en cuál de los dos mundos estamos.** Hoy no lo sabemos.

## 8.5 Reportar resultados incómodos

Entrenamos también con **3 000 imágenes** y salió **peor** que con 2 500 (0.885 vs. 0.914 en mAP@50). El paper no lo reporta; la curva termina en el mejor punto y la frase "2 500 imágenes son suficientes" parece más sólida de lo que es.

!!! warning "Reporte selectivo"
    Omitir resultados negativos o inconvenientes es una práctica cuestionable (*cherry-picking*). Un revisor atento que vea el repositorio lo notará. La corrección es sencilla: reportar el punto de 3 000, explicar la posible causa (otro test, ruido) y no concluir de más. Un resultado "incómodo" explicado honestamente da más credibilidad que una curva limpia.

## 8.6 Amenazas a la validez

| Tipo | Pregunta | Nuestro caso |
| --- | --- | --- |
| **Interna** | ¿Lo medido lo causó lo que creo? | confusores del §8.2; una sola semilla |
| **Externa** | ¿Se generaliza a otras escenas/ciudades/cámaras? | solo vistas cenitales de dron en Lima, sin análisis por hora/clima |
| **Constructo** | ¿La métrica mide lo que importa? | mAP no es conteo vehicular ni utilidad para tráfico |
| **Conclusión** | ¿Hay datos y estadística suficientes? | clases con 2–23 instancias de test; sin intervalos |

## 8.7 Auditoría de las afirmaciones del paper

Una forma práctica de revisar un artículo: tabular cada afirmación y preguntar si la evidencia la sostiene.

| Afirmación del paper | ¿Sostenida? | Por qué |
| --- | --- | --- |
| "Fine-tuning YOLO11n-OBB logra 93.9 % mAP@50 y 81.7 % mAP@50–95" | **Sí, para ese test** | verificado reproduciendo (dif. < 0.001); pero un solo split y sin IC |
| "2 500 imágenes bastan para un buen rendimiento" | **Débil** | el subconjunto de 3 000 es peor; test distinto por fila; sin semillas |
| "El escalado de 1 000 a 2 500 sube el mAP de 66.2 a 91.4 %" | **Confundida** | test distinto por tamaño; los clips casi no aumentan |
| "640 px / 70 ép. mejora 2.5 puntos" | **No aislada** | cambian resolución y épocas; el de 50 épocas no había convergido |
| "OBB adecuado para este dominio" | **No demostrada** | sin baseline HBB; 81 % de cajas alineadas a ejes |
| "3.33 FPS en Pi 4 (544 px) y 2.50 (640 px)" | **Plausible, no auditable** | el script se perdió; falta versión de NCNN y governor; 90–120 s |
| "No se observó throttling" | **Limitada** | flag final 0x0, pero la temperatura seguía subiendo |
| Motivación: "vistas oblicuas/cámaras elevadas" | **Incorrecta** | las imágenes son cenitales de dron |
| Figs. 2 y 3 como resultados finales | **Era débil; corregido** | usaban validación; ahora se generan sobre test |

## 8.8 Reproducibilidad

Reproducir = que otra persona, con tus datos y tu código, obtenga tus resultados. Mínimo a declarar y guardar:

- Versiones de **software** (Python, PyTorch, Ultralytics, NCNN) y de **hardware** (CPU/GPU, RAM, SO, governor).
- **Semillas** y `deterministic=True`.
- **Código y configuración** versionados (por eso existe `scripts/` y `results/`).
- **Datos**: cómo obtenerlos y cómo se construyeron los splits (el `train.csv` original no está en el repo: es una carencia).
- **Logs** del entrenamiento (aquí se perdió el log que diría qué optimizador eligió `auto`).

## 8.9 Cómo leer un paper

1. Lee **resumen y conclusiones** para saber qué afirma.
2. Mira **las tablas y figuras**: ¿qué se compara con qué? ¿hay barras de error?
3. Lee **el método** para ver qué se controló.
4. Haz la pregunta clave: **si la afirmación fuera falsa, ¿qué habría que ver diferente en los resultados?** Si nada, el experimento no la pone a prueba.
5. Busca lo que *no* está (baselines, ablaciones, resultados negativos, limitaciones).

## Resumen

- Cambia una cosa por vez; usa un test fijo; repite con semillas; incluye baselines.
- Reporta todo, también lo que no gusta.
- Revisa cada afirmación contra su evidencia.

!!! question "Comprueba que entendiste"
    1. ¿Qué dos factores cambian entre "544 px/50 ép." y "640 px/70 ép."?
    2. ¿Por qué un mismo modelo puede dar 0.86 y 0.91 en dos conjuntos de 376 imágenes?
    3. ¿Qué haría falta para afirmar que "OBB ayuda"?
    4. ¿Por qué se debe reportar el resultado de 3 000 imágenes?

    ??? success "Respuestas"
        1. Resolución y número de épocas.
        2. Las dos muestras contienen escenas distintas; la métrica depende de qué escenas caigan en cada una (varianza de muestreo).
        3. Una comparación con HBB sobre los mismos datos/splits/semillas, con intervalos de confianza.
        4. Porque omitirlo es reporte selectivo y debilita la conclusión "2 500 bastan"; explicar la causa fortalece el trabajo.

## Para profundizar

- Lipton & Steinhardt, *Troubling Trends in Machine Learning Scholarship*, 2018.
- Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018 (variabilidad entre semillas; aplica a cualquier ML).
- Pineau et al., *Improving Reproducibility in Machine Learning Research*, JMLR 2021 (lista de verificación).
