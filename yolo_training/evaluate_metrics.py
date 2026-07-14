import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
from ultralytics import YOLO

def main():
    # Ruta al mejor modelo guardado (se asume que terminó de entrenar en el run #1)
    # ultralytics guarda en runs/obb/yolo11n_obb_benchmark/weights/best.pt por defecto
    model_path = r'runs\obb\yolo11n_obb_benchmark\weights\best.pt'
    
    base_dir = r'runs'
    # Buscar el subdirectorio más reciente que comience con yolo11n_obb_benchmark
    possible_dirs = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            if d.startswith('yolo11n_obb_benchmark'):
                possible_dirs.append(os.path.join(base_dir, d))
    
    model_path = None
    if possible_dirs:
        latest_dir = sorted(possible_dirs, key=os.path.getmtime, reverse=True)[0]
        model_path = os.path.join(latest_dir, 'weights', 'best.pt')
                
    if not os.path.exists(model_path):
        print("El modelo no ha terminado de entrenarse o no se encontró best.pt")
        return

    print(f"Evaluando modelo: {model_path}")
    model = YOLO(model_path)
    
    # Evaluar en el conjunto de validación
    metrics = model.val(data='dataset.yaml')
    
    # Imprimir métricas clave
    print("\n--- Resultados del Benchmark ---")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    
    # Análisis de "Bias" (Clases más predecidas vs reales, u otro aspecto simple)
    # Como Ultralytics ya nos da Precision/Recall por clase, podemos ver si hay clases "olvidadas"
    class_names = metrics.names
    ap_per_class = metrics.box.maps # vector de AP por clase
    
    print("\n--- Análisis de Sesgo (Sesgo por Clase: Variación de mAP) ---")
    print("Las clases con mAP notablemente inferior pueden indicar un sesgo de los datos (e.g., falta de ejemplos de esa clase o casos difíciles)")
    
    for c_id, map_val in enumerate(ap_per_class):
        c_name = class_names[c_id] if c_id in class_names else str(c_id)
        print(f"Clase '{c_name}': mAP@50-95 = {map_val:.4f}")

    # Guardar métricas completas en JSON para fácil acceso
    results_json = r'benchmark_results.json'
    with open(results_json, 'w') as f:
        json.dump(metrics.results_dict, f, indent=4)
        
    print(f"\nMétricas detalladas guardadas en {results_json}")

if __name__ == '__main__':
    main()
