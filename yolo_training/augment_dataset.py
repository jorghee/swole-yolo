import os
import shutil
from collections import Counter

def main():
    images_dir = r'c:\IA\yolo_obb_dataset\images\train'
    labels_dir = r'c:\IA\yolo_obb_dataset\labels\train'
    
    # Multiplicadores para cada clase minoritaria (indice de clase YOLO)
    # basándonos en la distribución que medimos:
    multipliers = {
        5: 20, # class6: ~12 instancias -> 20x
        2: 10, # class3: ~111 instancias -> 10x
        4: 10, # class5: ~106 instancias -> 10x
        7: 5,  # class8: ~233 instancias -> 5x
        1: 5,  # class2: ~465 instancias -> 5x
        3: 5,  # class4: ~838 instancias -> 5x
        6: 3,  # class7: ~1433 instancias -> 3x
        8: 3,  # class9: ~2088 instancias -> 3x
    }
    
    files_to_duplicate = {}
    
    # Escanear archivos y determinar cuantas veces hay que duplicarlos
    for f in os.listdir(labels_dir):
        if not f.endswith('.txt') or '_aug' in f:
            continue
            
        filepath = os.path.join(labels_dir, f)
        
        # Encontrar la clase minoritaria más rara en este archivo
        max_multiplier = 0
        with open(filepath, 'r') as file:
            for line in file:
                cls_id = int(line.split()[0])
                if cls_id in multipliers:
                    max_multiplier = max(max_multiplier, multipliers[cls_id])
                    
        if max_multiplier > 0:
            # Quitamos 1 porque el original ya existe
            files_to_duplicate[f] = max_multiplier - 1
            
    # Realizar la duplicación
    print(f"Duplicando {len(files_to_duplicate)} archivos...")
    for label_file, times in files_to_duplicate.items():
        base_name = label_file.replace('.txt', '')
        
        orig_label_path = os.path.join(labels_dir, label_file)
        orig_img_path = os.path.join(images_dir, base_name + '.jpg')
        # En el dataset original las imágenes pueden ser png o jpg, revisamos:
        if not os.path.exists(orig_img_path):
            orig_img_path = os.path.join(images_dir, base_name + '.png')
            if not os.path.exists(orig_img_path):
                print(f"No se encontró imagen para {label_file}")
                continue
                
        img_ext = os.path.splitext(orig_img_path)[1]
                
        for i in range(times):
            new_base = f"{base_name}_aug{i}"
            new_label_path = os.path.join(labels_dir, f"{new_base}.txt")
            new_img_path = os.path.join(images_dir, f"{new_base}{img_ext}")
            
            shutil.copy2(orig_label_path, new_label_path)
            shutil.copy2(orig_img_path, new_img_path)
            
    print("¡Sobremuestreo completado con éxito!")

if __name__ == '__main__':
    main()
