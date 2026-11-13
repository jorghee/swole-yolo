import psutil
import time
import os
import ctypes

# Constantes de Windows para prevenir la suspensión
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def vigilar_y_apagar():
    # Le dice a Windows: "¡No te suspendas, estoy trabajando!"
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    
    print("Vigilando el proceso de entrenamiento de YOLO...")
    print("Se ha bloqueado la suspensión automática de Windows temporalmente.")
    
    while True:
        entrenamiento_activo = False
        
        # Revisar todos los procesos activos en la computadora
        for p in psutil.process_iter(['name', 'cmdline']):
            try:
                # Si es Python y está ejecutando train.py
                if p.info['name'] and 'python' in p.info['name'].lower() and p.info['cmdline']:
                    if 'train.py' in ' '.join(p.info['cmdline']):
                        entrenamiento_activo = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if not entrenamiento_activo:
            print("\n¡El entrenamiento ha terminado (o se cerró)!")
            print("Apagando la PC en 60 segundos...")
            print("Para cancelar el apagado, abre una terminal y escribe: shutdown /a")
            # Ejecuta el comando de Windows para apagar la PC en 60 segundos
            os.system("shutdown /s /t 60")
            break
            
        # Esperar 1 minuto antes de volver a revisar
        time.sleep(60)

if __name__ == '__main__':
    vigilar_y_apagar()
