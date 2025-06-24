import cv2
import os
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from threading import Thread
import time

# === CONFIGURACIÓN ===
SEQUENCE_LENGTH = 150
INPUT_DIM = 144
CLASES = [
    "sentadilla", "flexion", "flexion_codo", "estiramiento",
    "sentadilla_incorrecta", "flexion_incorrecta", "flexion_codo_incorrecta", "estiramiento_incorrecta"
]
DATA_DIR = "data/dataset"

# === MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# === Ángulos útiles ===
def calcular_angulos(landmarks):
    def get_angle(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)
                                      * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    lmk = landmarks.landmark
    return [
        get_angle(lmk[23], lmk[25], lmk[27]),  # pierna izq
        get_angle(lmk[24], lmk[26], lmk[28]),  # pierna der
        get_angle(lmk[11], lmk[13], lmk[15]),  # brazo izq
        get_angle(lmk[12], lmk[14], lmk[16]),  # brazo der
        get_angle(lmk[11], lmk[23], lmk[25]),  # torso izq
        get_angle(lmk[12], lmk[24], lmk[26])   # torso der
    ]


# === Interfaz ===
class CapturaGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Captura de Ejercicios (150 frames)")
        self.ejercicio = tk.StringVar(value=CLASES[0])
        self.capturando = False
        self.repeticiones = 0

        ttk.Label(self.root, text="Selecciona clase:").pack(pady=5)
        self.selector = ttk.Combobox(
            self.root, values=CLASES, textvariable=self.ejercicio)
        self.selector.pack(pady=5)

        self.start_btn = ttk.Button(
            self.root, text="Iniciar captura", command=self.iniciar)
        self.start_btn.pack(pady=5)

        self.stop_btn = ttk.Button(
            self.root, text="Detener", command=self.detener, state="disabled")
        self.stop_btn.pack(pady=5)

        self.status_lbl = ttk.Label(self.root, text="Estado: Esperando...")
        self.status_lbl.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.salir)

        self.thread = None
        self.root.mainloop()

    def iniciar(self):
        self.capturando = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_lbl.config(text="Capturando...")
        self.thread = Thread(target=self.capturar)
        self.thread.start()

    def detener(self):
        self.capturando = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_lbl.config(text="Estado: Detenido")

    def salir(self):
        self.capturando = False
        self.root.destroy()

    def capturar(self):
        cap = cv2.VideoCapture(0)
        secuencia = []
        while self.capturando:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                puntos = results.pose_landmarks.landmark
                keypoints = []
                for lm in puntos[:33]:
                    keypoints.extend([lm.x, lm.y])
                angulos = calcular_angulos(results.pose_landmarks)
                if len(keypoints) == 66 and len(angulos) == 6:
                    caracteristicas = keypoints + angulos
                    caracteristicas += [0.0] * \
                        (INPUT_DIM - len(caracteristicas))
                    secuencia.append(caracteristicas)

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Frames: {len(secuencia)}/{SEQUENCE_LENGTH}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Captura", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(secuencia) == SEQUENCE_LENGTH:
                etiqueta = self.ejercicio.get()
                carpeta = os.path.join(DATA_DIR, etiqueta)
                os.makedirs(carpeta, exist_ok=True)
                nombre_archivo = f"rep_{self.repeticiones+1:03d}.npy"
                ruta = os.path.join(carpeta, nombre_archivo)
                np.save(ruta, np.array(secuencia))
                self.repeticiones += 1
                self.status_lbl.config(
                    text=f"Repetición {self.repeticiones} guardada.")
                secuencia = []

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CapturaGUI()
