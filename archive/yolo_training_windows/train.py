import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

def main():
    # Cargar modelo YOLO11n-OBB preentrenado
    model = YOLO('yolo11n-obb.pt')

    # Iniciar entrenamiento
    results = model.train(
        data='dataset.yaml',
        epochs=50,
        fraction=0.25,   # Usa el 25% del dataset por época. ¡Reduce el tiempo de 25h a ~6h!
        imgsz=640,       # (Opcional: bajar a 480 para ir más rápido, pero sacrifica algo de precisión)
        project='runs',
        name='yolo11n_obb_fast_train',
        device=0,
        batch=16,        # Si bajas imgsz a 480, podrías subir esto a 32.
        workers=8,       # Aumentado a 8 para que la CPU envíe datos más rápido a la GPU
        patience=10,     # Si en 10 epochs no mejora, detiene el entrenamiento (ahorra tiempo)
        save_period=5,   # Guarda los pesos cada 5 epochs por si la PC se apaga o falla
        mosaic=1.0,      # Aumentado para ver imágenes compuestas
        mixup=0.2,       # Aumentado ligeramente
        degrees=45.0,    # Permitir rotaciones para OBB
        shear=5.0,
        flipud=0.5,
        fliplr=0.5,
    )

if __name__ == '__main__':
    main()
