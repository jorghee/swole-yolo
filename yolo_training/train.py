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
        imgsz=640,
        project='runs',
        name='yolo11n_obb_benchmark_aug',
        device=0,
        mosaic=1.0,      # Aumentado para ver imágenes compuestas
        mixup=0.2,       # Aumentado ligeramente
        degrees=45.0,    # Permitir rotaciones para OBB
        shear=5.0,
        flipud=0.5,
        fliplr=0.5,
    )

if __name__ == '__main__':
    main()
