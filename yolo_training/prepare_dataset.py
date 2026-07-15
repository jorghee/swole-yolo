import os
import csv
import math
import random
import shutil

# Rutas
csv_path = r'c:\IA\mtc_challenge-20260630T032548Z-3-003\mtc_challenge\train.csv'
images_src_dir = r'c:\IA\train-001\train'
dataset_dest_dir = r'c:\IA\yolo_obb_dataset_full'

# Crear estructura
os.makedirs(os.path.join(dataset_dest_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dest_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(dataset_dest_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dest_dir, 'labels', 'val'), exist_ok=True)

# Parámetros de imagen
IMG_W = 1920
IMG_H = 1080

def cxcywha_to_corners(cx, cy, w, h, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    dx = w / 2
    dy = h / 2
    
    # 4 esquinas de la caja orientada
    corners = [
        (-dx, -dy),
        ( dx, -dy),
        ( dx,  dy),
        (-dx,  dy)
    ]
    
    rotated_corners = []
    for x, y in corners:
        # Rotación estándar en 2D
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        
        # Normalizar
        rx_norm = rx / IMG_W
        ry_norm = ry / IMG_H
        rotated_corners.extend([rx_norm, ry_norm])
        
    return rotated_corners

# Leer CSV y separar train/val
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader) # Saltar cabecera
    rows = list(reader)

from collections import defaultdict
video_to_rows = defaultdict(list)
for row in rows:
    img_id = row[0]
    # Extraer el ID del video (ej: v_ab12cd34ef)
    vid_id = img_id.rsplit('_', 1)[0]
    video_to_rows[vid_id].append(row)

unique_vids = list(video_to_rows.keys())
random.seed(42)
random.shuffle(unique_vids)

# Separar el 80% de los VIDEOS (no de los frames individuales) para evitar data leakage
split_idx = int(len(unique_vids) * 0.8)
train_vids = unique_vids[:split_idx]
val_vids = unique_vids[split_idx:]

train_rows = []
for vid in train_vids:
    train_rows.extend(video_to_rows[vid])
    
val_rows = []
for vid in val_vids:
    val_rows.extend(video_to_rows[vid])

def process_split(split_rows, split_name):
    for row in split_rows:
        img_id = row[0]
        target = row[1]
        
        img_filename = f"{img_id}.jpg"
        src_img_path = os.path.join(images_src_dir, img_filename)
        dest_img_path = os.path.join(dataset_dest_dir, 'images', split_name, img_filename)
        
        if not os.path.exists(src_img_path):
            continue
            
        shutil.copy2(src_img_path, dest_img_path)
        
        dest_lbl_path = os.path.join(dataset_dest_dir, 'labels', split_name, f"{img_id}.txt")
        
        with open(dest_lbl_path, 'w') as lf:
            if target != 'none':
                objects = target.split(';')
                for obj in objects:
                    parts = obj.split()
                    if len(parts) >= 6:
                        # Clases son 1-9, restamos 1 para 0-8
                        c = int(parts[0]) - 1
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        angle = float(parts[5])
                        
                        corners = cxcywha_to_corners(cx, cy, w, h, angle)
                        corners_str = ' '.join(f"{v:.6f}" for v in corners)
                        lf.write(f"{c} {corners_str}\n")

print("Procesando datos de entrenamiento...")
process_split(train_rows, 'train')
print("Procesando datos de validación...")
process_split(val_rows, 'val')
print("Dataset OBB preparado correctamente.")
