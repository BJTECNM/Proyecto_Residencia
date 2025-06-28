import os
import cv2
import numpy as np
import mediapipe as mp
from math import acos, degrees

# === Configuración ===
INPUT_DIR = "data/preparar/de_pie_05"
OUTPUT_FILE = "data/dataset/de_pie/rep_05.npy"
SEQUENCE_LENGTH = 60
POINTS = 33
VALORES_POR_FRAME = 99  # x, y, visibilidad

# === Inicializa MediaPipe ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def calcular_angulo(a, b, c):
    """Calcula el ángulo entre tres puntos (en 2D)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angulo = np.arccos(np.clip(cos_angulo, -1.0, 1.0))
    return degrees(angulo)

def procesar_imagen(imagen):
    results = pose.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return np.zeros(VALORES_POR_FRAME)

    puntos = results.pose_landmarks.landmark

    # Coordenadas relativas y normalización por altura
    base = puntos[23]  # cadera izquierda
    hombro = puntos[11]
    tobillo = puntos[27]
    altura = np.linalg.norm([hombro.x - tobillo.x, hombro.y - tobillo.y]) + 1e-6

    datos = []
    for punto in puntos:
        x_rel = (punto.x - base.x) / altura
        y_rel = (punto.y - base.y) / altura
        datos.extend([x_rel, y_rel, punto.visibility])

    # 6 ángulos adicionales (en grados)
    try:
        ang_codo_der = calcular_angulo(
            [puntos[12].x, puntos[12].y],  # hombro der
            [puntos[14].x, puntos[14].y],  # codo der
            [puntos[16].x, puntos[16].y])  # muñeca der

        ang_codo_izq = calcular_angulo(
            [puntos[11].x, puntos[11].y],
            [puntos[13].x, puntos[13].y],
            [puntos[15].x, puntos[15].y])

        ang_rodilla_der = calcular_angulo(
            [puntos[24].x, puntos[24].y],
            [puntos[26].x, puntos[26].y],
            [puntos[28].x, puntos[28].y])

        ang_rodilla_izq = calcular_angulo(
            [puntos[23].x, puntos[23].y],
            [puntos[25].x, puntos[25].y],
            [puntos[27].x, puntos[27].y])

        ang_hombros = calcular_angulo(
            [puntos[11].x, puntos[11].y],
            [(puntos[11].x + puntos[12].x) / 2, (puntos[11].y + puntos[12].y) / 2],
            [puntos[12].x, puntos[12].y])

        ang_caderas = calcular_angulo(
            [puntos[23].x, puntos[23].y],
            [(puntos[23].x + puntos[24].x) / 2, (puntos[23].y + puntos[24].y) / 2],
            [puntos[24].x, puntos[24].y])

        datos.extend([
            ang_codo_der, ang_codo_izq,
            ang_rodilla_der, ang_rodilla_izq,
            ang_hombros, ang_caderas
        ])
    except:
        datos.extend([0.0] * 6)  # Si fallan los cálculos, rellenar con ceros

    return np.array(datos)

# === Procesar imágenes ===
secuencia = []

for i in range(1, SEQUENCE_LENGTH + 1):
    nombre = f"f{i}.jpg"
    ruta = os.path.join(INPUT_DIR, nombre)
    imagen = cv2.imread(ruta)
    if imagen is None:
        print(f"⚠️ Imagen no encontrada: {nombre}, usando ceros.")
        secuencia.append(np.zeros(VALORES_POR_FRAME + 6))
    else:
        vector = procesar_imagen(imagen)
        secuencia.append(vector)

# === Guardar .npy ===
secuencia = np.array(secuencia)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
np.save(OUTPUT_FILE, secuencia)
print(f"✅ Secuencia guardada en: {OUTPUT_FILE}")
