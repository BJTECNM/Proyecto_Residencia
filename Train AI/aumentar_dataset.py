import os
import numpy as np
import glob

# === Transformaciones ===

def rotar_frame_2d(frame, angulo_grados):
    ang_rad = np.radians(angulo_grados)
    cos_a, sin_a = np.cos(ang_rad), np.sin(ang_rad)
    rotado = []
    for i in range(33):
        x = frame[i * 3]
        y = frame[i * 3 + 1]
        v = frame[i * 3 + 2]
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        rotado.extend([x_rot, y_rot, v])
    return np.array(rotado)

def agregar_ruido(frame, intensidad=0.01):
    ruido = np.random.normal(0, intensidad, size=frame.shape)
    return frame + ruido

def escalar_frame(frame, factor=1.05):
    escalado = []
    for i in range(33):
        x = frame[i * 3] * factor
        y = frame[i * 3 + 1] * factor
        v = frame[i * 3 + 2]
        escalado.extend([x, y, v])
    return np.array(escalado)

def flip_horizontal(frame):
    volteado = []
    for i in range(33):
        x = -frame[i * 3]
        y = frame[i * 3 + 1]
        v = frame[i * 3 + 2]
        volteado.extend([x, y, v])
    return np.array(volteado)

def aplicar_aumento(secuencia, tipo):
    if tipo == "rotp":
        return np.array([rotar_frame_2d(f, +5) for f in secuencia])
    elif tipo == "rotm":
        return np.array([rotar_frame_2d(f, -5) for f in secuencia])
    elif tipo == "noise":
        return np.array([agregar_ruido(f) for f in secuencia])
    elif tipo == "flip":
        return np.array([flip_horizontal(f) for f in secuencia])
    elif tipo == "scale":
        return np.array([escalar_frame(f) for f in secuencia])
    else:
        raise ValueError("Tipo de aumento desconocido")

# === Configuración general ===
DATASET_PATH = "data/dataset"
TIPOS_AUMENTO = ["rotp", "rotm", "noise", "flip", "scale"]

# === Recorrer carpetas y aplicar aumentos ===
total_generados = 0

for clase in os.listdir(DATASET_PATH):
    clase_path = os.path.join(DATASET_PATH, clase)
    if not os.path.isdir(clase_path):
        continue

    archivos = glob.glob(os.path.join(clase_path, "*.npy"))
    for archivo in archivos:
        base = os.path.splitext(os.path.basename(archivo))[0]
        secuencia = np.load(archivo)

        if secuencia.shape[1] != 99:
            print(f"❌ Ignorado (shape incompatible): {archivo}")
            continue

        for tipo in TIPOS_AUMENTO:
            aug = aplicar_aumento(secuencia, tipo)
            nuevo_nombre = f"{base}_{tipo}.npy"
            nueva_ruta = os.path.join(clase_path, nuevo_nombre)
            np.save(nueva_ruta, aug)
            total_generados += 1

print(f"✅ {total_generados} secuencias aumentadas generadas.")
