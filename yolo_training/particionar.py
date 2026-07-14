import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm

# ==========================================
# CONFIGURACIÓN
# ==========================================
CSV_ORIGINAL = r"c:\IA\mtc_challenge-20260630T032548Z-3-003\mtc_challenge\train.csv"        # Tu CSV con las 54,263 líneas
DIR_IMAGENES = r"c:\IA\train-001\train\\"            # Carpeta con tus 54,263 imágenes
DIR_SALIDA = r"c:\IA\datasets_porciones\\"  # Carpeta principal de salida
PORCIONES = [1000, 1500, 2000, 2500, 3000]     # Tamaños de los subsets a generar
EXTENSION = ".jpg"                  # Ajusta a .png si es necesario
USAR_SYMLINKS = False               # False = copia los archivos fisicamente (requerido para subir a Drive)
# ==========================================

def seleccionar_frames_distanciados(df_grupo, cantidad):
    """Selecciona 'cantidad' de frames lo más separados posible en la línea de tiempo."""
    if cantidad >= len(df_grupo):
        return df_grupo
    
    # Ordenar por número de frame (0 al 49)
    df_grupo = df_grupo.sort_values('Frame_Num')
    # np.linspace selecciona índices matemáticamente separados (ej. inicio, medio, fin)
    indices = np.linspace(0, len(df_grupo) - 1, num=cantidad, dtype=int)
    return df_grupo.iloc[indices]

def extraer_porciones_dinamicas(df, tamaño_objetivo):
    """Calcula la cuota por video y extrae los frames correspondientes."""
    print(f"\n--- Calculando distribución para subset de {tamaño_objetivo} ---")
    
    # 1. Separar Video_ID y Frame_Num (ej: v_009evckk5b_0049 -> ID: v_009evckk5b, Num: 49)
    df['Video_ID'] = df['Id'].apply(lambda x: x.rsplit('_', 1)[0])
    df['Frame_Num'] = df['Id'].apply(lambda x: int(x.rsplit('_', 1)[1]))
    
    videos_unicos = df['Video_ID'].unique()
    total_videos = len(videos_unicos)
    
    if tamaño_objetivo > len(df):
        print(f"Error: El dataset original ({len(df)}) es menor al objetivo.")
        return pd.DataFrame()
        
    # 2. Calcular cuotas (división entera y el residuo)
    frames_base = tamaño_objetivo // total_videos
    frames_sobrantes = tamaño_objetivo % total_videos
    
    print(f"Videos únicos detectados: {total_videos}. Asignando {frames_base} frame(s) base por video...")
    
    # 3. Asignar las cuotas a cada video (repartiendo el residuo entre los primeros N videos)
    cuotas = {vid: frames_base for vid in videos_unicos}
    for vid in videos_unicos[:frames_sobrantes]: 
        cuotas[vid] += 1
        
    # 4. Extraer los frames espaciados
    dfs_seleccionados = []
    for vid, grupo_video in df.groupby('Video_ID'):
        cuota = cuotas[vid]
        if cuota > 0:
            seleccion = seleccionar_frames_distanciados(grupo_video, cuota)
            dfs_seleccionados.append(seleccion)
            
    df_final = pd.concat(dfs_seleccionados)
    
    # Limpieza de columnas temporales
    df_final = df_final.drop(columns=['Video_ID', 'Frame_Num'])
    
    print(f"Subset generado con éxito: {len(df_final)} imágenes.")
    return df_final

def crear_porciones():
    print(f"Cargando {CSV_ORIGINAL}...")
    df_completo = pd.read_csv(CSV_ORIGINAL)
    
    os.makedirs(DIR_SALIDA, exist_ok=True)
    
    for tamaño in PORCIONES:
        df_muestra = extraer_porciones_dinamicas(df_completo.copy(), tamaño)
        
        if df_muestra.empty:
            continue
            
        nombre_porcion = f"dataset_{tamaño}"
        ruta_porcion = os.path.join(DIR_SALIDA, nombre_porcion)
        ruta_img_porcion = os.path.join(ruta_porcion, "images")
        
        os.makedirs(ruta_img_porcion, exist_ok=True)
        
        # Guardar el CSV particionado
        ruta_csv_nuevo = os.path.join(ruta_porcion, f"etiquetas_{tamaño}.csv")
        df_muestra.to_csv(ruta_csv_nuevo, index=False)
        
        # Crear Symlinks o copiar físicamente
        errores = 0
        for img_id in tqdm(df_muestra['Id'], desc=f"Procesando imágenes {tamaño}"):
            nombre_archivo = f"{img_id}{EXTENSION}"
            ruta_origen = os.path.join(DIR_IMAGENES, nombre_archivo)
            ruta_destino = os.path.join(ruta_img_porcion, nombre_archivo)
            
            if not os.path.exists(ruta_origen):
                errores += 1
                continue
                
            if not os.path.exists(ruta_destino):
                if USAR_SYMLINKS:
                    os.symlink(os.path.abspath(ruta_origen), os.path.abspath(ruta_destino))
                else:
                    shutil.copy2(ruta_origen, ruta_destino)
                    
        if errores > 0:
            print(f"Advertencia: No se encontraron {errores} imágenes físicamente en el disco.")
            
    print("\n¡Particionamiento dinámico completado con éxito!")

if __name__ == "__main__":
    crear_porciones()