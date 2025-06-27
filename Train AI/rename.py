import os

carpeta = "data/dataset/sentadilla_06"
archivos = sorted(os.listdir(carpeta))
for i, nombre in enumerate(archivos[:60], start=1):
    origen = os.path.join(carpeta, nombre)
    destino = os.path.join(carpeta, f"f{i}.jpg")
    os.rename(origen, destino)

print("✅ Archivos renombrados del 1 al 60 como f1.jpg, f2.jpg...")
