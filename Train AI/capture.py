import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import math

# Nombre del ejercicio a capturar
EXERCISE_LABEL = "sentadilla"
CSV_FILE = "modelo_ejercicios.csv"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calcular_angulo(a, b, c):
    """Calcula el ángulo entre tres puntos"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angulo = np.arccos(np.clip(cos_angulo, -1.0, 1.0))
    return np.degrees(angulo)

def extraer_angulos(landmarks):
    puntos = [(lm.x, lm.y) for lm in landmarks]

    # Elegimos algunos ángulos clave
    angulos = []
    angulos.append(calcular_angulo(puntos[11], puntos[13], puntos[15]))  # Brazo izq
    angulos.append(calcular_angulo(puntos[12], puntos[14], puntos[16]))  # Brazo der
    angulos.append(calcular_angulo(puntos[23], puntos[25], puntos[27]))  # Pierna izq
    angulos.append(calcular_angulo(puntos[24], puntos[26], puntos[28]))  # Pierna der
    angulos.append(calcular_angulo(puntos[25], puntos[27], puntos[31]))  # Rodilla izq
    angulos.append(calcular_angulo(puntos[26], puntos[28], puntos[32]))  # Rodilla der
    return angulos

# CSV setup
file_exists = os.path.isfile(CSV_FILE)
csv_file = open(CSV_FILE, 'a', newline='')
csv_writer = csv.writer(csv_file)
if not file_exists:
    header = [f'ang_{i}' for i in range(6)] + ['label']
    csv_writer.writerow(header)

cap = cv2.VideoCapture(0)
print("[INFO] Presiona 's' para guardar muestra, 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Captura de Ejercicio", image)
    key = cv2.waitKey(1)

    if key == ord('s') and results.pose_landmarks:
        angulos = extraer_angulos(results.pose_landmarks.landmark)
        angulos.append(EXERCISE_LABEL)
        csv_writer.writerow(angulos)
        print("[INFO] Muestra guardada.")

    elif key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
