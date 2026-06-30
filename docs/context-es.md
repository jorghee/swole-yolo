# Implementacion de un modelo para la deteccion de 9 clases de vehiculos

## Sources

- data/processed_300/data_300.csv: csv con las imagenes y las bounding boxes
- data/processed_300/imgs/: imagenes

## Objetivo

El objetivo principal es desarrollar una solución basada en inteligencia artificial capaz de detectar y clasificar vehículos en intersecciones urbanas a partir de imagenes

## CSV Structure

El archivo train.csv contiene las anotaciones de entrenamiento.

### Tiene dos columnas

Id,Target
Donde:

- Id es el nombre del archivo sin extensión.
- Target contiene las anotaciones del frame correspondiente.
Por ejemplo, si en train.zip existe la imagen v_ab12cd34ef_0000.jpg, entonces en train.csv el identificador correspondiente será v_ab12cd34ef_0000.

### Formato de la columna Target en train.csv

Cada objeto anotado se representa como una caja orientada usando el siguiente formato:

category_id cx cy width height angle_deg
Donde:

category_id es el identificador de la clase.
cx es la coordenada x del centro de la caja.
cy es la coordenada y del centro de la caja.
width es el ancho de la caja.
height es el alto de la caja.
angle_deg es el ángulo de rotación de la caja, en grados.
Todas las coordenadas están expresadas en píxeles. Si un frame contiene varios objetos, estos se separan con punto y coma ;.
(pueden haber mas de una annotacion en un frame o ninguna representada por 'none' en el campo Target)

Ejemplo:

Id,Target
v_ab12cd34ef_0000,"1 987.86 598.84 48.84 94.88 339.94;9 1236.10 506.05 39.07 29.30 0.00"
Si el modelo no predice ningún objeto para un frame, el campo Target debe contener none. No se aceptan celdas vacías en la columna Target.

Clases Oficiales
Las categorías oficiales son:

ID 1: Auto
ID 2: Combi
ID 3: microbus
ID 4: minibus
ID 5: omnibus
ID 6: articulado
ID 7: camion
ID 8: mototaxi
ID 9: motocicleta

### Formato de Predicción

Para cada frame del conjunto de prueba, el participante deberá enviar un conjunto de detecciones en la columna Target.

Cada predicción debe tener el siguiente formato:

score category_id cx cy width height angle_deg
donde:

score: confianza de la detección, entre 0 y 1.
category_id: identificador de clase, entre 1 y 9.
cx: coordenada x del centro de la caja predicha.
cy: coordenada y del centro de la caja predicha.
width: ancho de la caja predicha.
height: alto de la caja predicha.
angle_deg: ángulo de rotación de la caja predicha, en grados.
Varias detecciones para un mismo frame deben separarse usando punto y coma ;.

Ejemplo:

0.93 1 987.86 598.84 48.84 94.88 339.94;0.81 9 1236.10 506.05 39.07 29.30 0.00
Si el modelo no predice ningún objeto para un frame, el campo Target debe contener exactamente none. No se aceptan celdas vacías en la columna Target.
