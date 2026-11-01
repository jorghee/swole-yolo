# YOLO OBB Training Repository

Este repositorio contiene los scripts necesarios para procesar el dataset, aumentarlo, entrenar un modelo YOLO11n-OBB y evaluarlo.

## Requisitos

Para instalar todas las dependencias necesarias, ejecuta:
```bash
pip install -r requirements.txt
```

## Flujo de Trabajo

El flujo completo para preparar los datos y entrenar el modelo es el siguiente:

1. **Particionar el Dataset**
   Extrae porciones balanceadas de imágenes del dataset original.
   ```bash
   python particionar.py
   ```

2. **Preparar el Dataset en formato YOLO OBB**
   Toma la porción extraída (e.g., de 3000 imágenes) y crea la estructura `images/train`, `images/val`, `labels/train` y `labels/val`, además de convertir las coordenadas.
   ```bash
   python prepare_dataset.py
   ```

3. **Verificar Balance de Clases**
   Puedes verificar cuántas instancias de cada clase existen.
   ```bash
   python check_balance.py
   ```

4. **Aumentar las Clases Minoritarias (Opcional)**
   Duplica archivos de entrenamiento de las clases menos representadas para balancear el dataset.
   ```bash
   python augment_dataset.py
   ```

5. **Entrenar el Modelo YOLO**
   Inicia el entrenamiento del modelo usando los datos de `c:/IA/yolo_obb_dataset`.
   ```bash
   python train.py
   ```
   *Nota: Si tienes problemas de memoria o quieres correr en CPU, asegúrate de ajustar el parámetro `device` o el `batch_size` en `train.py`.*

6. **Evaluar Métricas**
   Una vez entrenado, puedes evaluar el rendimiento del mejor modelo guardado en la carpeta `runs`.
   ```bash
   python evaluate_metrics.py
   ```
