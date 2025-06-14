import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import time
import tkinter as tk
from tkinter import ttk
import pygame
from datetime import datetime

# ---------- Configuración general ----------
DATA_DIR = "data"
EXERCISES = {
    "Flexión codo": "flexion_codo",
    "Flexiones": "flexiones",
    "Sentadilla": "sentadilla",
    "Estiramiento": "estiramiento"
}
pygame.mixer.init()

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Variables globales
grabando = False
repeticion = []
start_time = None
exercise_label = ""


# ---------- Funciones auxiliares ----------
def reproducir_sonido(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angulo = np.dot(ba, bc) / (np.linalg.norm(ba)
                                   * np.linalg.norm(bc) + 1e-6)
    angulo = np.arccos(np.clip(cos_angulo, -1.0, 1.0))
    return np.degrees(angulo)


def extraer_datos(landmarks):
    puntos = [(lm.x, lm.y) for lm in landmarks]
    angulos = [
        calcular_angulo(puntos[11], puntos[13], puntos[15]),  # Brazo izq
        calcular_angulo(puntos[12], puntos[14], puntos[16]),  # Brazo der
        calcular_angulo(puntos[23], puntos[25], puntos[27]),  # Pierna izq
        calcular_angulo(puntos[24], puntos[26], puntos[28]),  # Pierna der
        calcular_angulo(puntos[25], puntos[27], puntos[31]),  # Rodilla izq
        calcular_angulo(puntos[26], puntos[28], puntos[32]),  # Rodilla der
    ]
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y])
    return angulos + coords


def guardar_repeticion():
    global repeticion, exercise_label, start_time

    if not repeticion:
        return

    end_time = time.time()
    duracion = round(end_time - start_time, 2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ejercicio_dir = os.path.join(DATA_DIR, exercise_label)

    os.makedirs(ejercicio_dir, exist_ok=True)
    rep_id = f"{exercise_label}_{timestamp}"

    npy_path = os.path.join(ejercicio_dir, f"{rep_id}.npy")
    np.save(npy_path, np.array(repeticion))

    # También guardamos CSV opcional (para depuración o entrenamiento clásico)
    csv_path = os.path.join(ejercicio_dir, f"{rep_id}.csv")
    with open(csv_path, 'w') as f:
        for frame in repeticion:
            f.write(",".join(map(str, frame)) + "\n")

    print(f"[INFO] Repetición guardada como {rep_id} ({duracion} s)")


# ---------- Interfaz gráfica ----------
def iniciar_grabacion():
    global grabando, repeticion, exercise_label, start_time

    exercise_label = EXERCISES[combo.get()]
    estado.set("Preparándote...")
    ventana.update()

    threading.Thread(target=esperar_y_empezar).start()


def esperar_y_empezar():
    global grabando, repeticion, start_time
    time.sleep(5)
    reproducir_sonido("start.mp3")
    repeticion = []
    grabando = True
    start_time = time.time()
    estado.set("Grabando...")
    ventana.update()


def detener_grabacion():
    global grabando
    grabando = False
    guardar_repeticion()
    estado.set("Listo.")
    ventana.update()


# ---------- Crear ventana ----------
ventana = tk.Tk()
ventana.title("Captura de Ejercicio")
ventana.geometry("400x200")

ttk.Label(ventana, text="Selecciona el ejercicio:").pack(pady=10)
combo = ttk.Combobox(ventana, values=list(EXERCISES.keys()), state="readonly")
combo.set("Sentadilla")
combo.pack()

ttk.Button(ventana, text="Iniciar Grabación",
           command=iniciar_grabacion).pack(pady=10)
ttk.Button(ventana, text="Detener Grabación",
           command=detener_grabacion).pack(pady=5)

estado = tk.StringVar()
estado.set("Listo.")
ttk.Label(ventana, textvariable=estado).pack(pady=10)


# ---------- Captura de cámara ----------
def procesar_video():
    global grabando, repeticion

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if grabando:
                datos = extraer_datos(results.pose_landmarks.landmark)
                repeticion.append(datos)

        cv2.imshow("Vista en vivo", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Ejecutar procesamiento de video en hilo separado
video_thread = threading.Thread(target=procesar_video)
video_thread.start()

# Iniciar interfaz
ventana.mainloop()
