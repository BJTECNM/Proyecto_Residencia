import tkinter as tk
from tkinter import ttk
import cv2
import os
import threading
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk

# Configuración general
FRAMES_POR_REPETICION = 60
DATASET_DIR = "data/dataset"

# Inicializa MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils


# Función para calcular ángulos adicionales
def calcular_angulos(landmarks):
    def angulo(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        ba = a - b
        bc = c - b
        cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba)
                                    * np.linalg.norm(bc) + 1e-6)
        ang = np.arccos(np.clip(cos_ang, -1.0, 1.0))
        return np.degrees(ang)

    p = mp_pose.PoseLandmark
    angles = []
    pairs = [
        (p.RIGHT_SHOULDER, p.RIGHT_ELBOW, p.RIGHT_WRIST),
        (p.LEFT_SHOULDER, p.LEFT_ELBOW, p.LEFT_WRIST),
        (p.RIGHT_HIP, p.RIGHT_KNEE, p.RIGHT_ANKLE),
        (p.LEFT_HIP, p.LEFT_KNEE, p.LEFT_ANKLE),
        (p.RIGHT_ELBOW, p.RIGHT_SHOULDER, p.RIGHT_HIP),
        (p.LEFT_ELBOW, p.LEFT_SHOULDER, p.LEFT_HIP),
        (p.RIGHT_SHOULDER, p.RIGHT_HIP, p.RIGHT_KNEE),
        (p.LEFT_SHOULDER, p.LEFT_HIP, p.LEFT_KNEE),
        (p.LEFT_HIP, p.NOSE, p.RIGHT_HIP),
        (p.LEFT_SHOULDER, p.NOSE, p.RIGHT_SHOULDER),
        (p.LEFT_KNEE, p.LEFT_HIP, p.RIGHT_HIP),
        (p.RIGHT_KNEE, p.RIGHT_HIP, p.LEFT_HIP)
    ]
    for a, b, c in pairs:
        angles.append(angulo(landmarks[a.value],
                      landmarks[b.value], landmarks[c.value]))
    return angles


# Clase principal
class CapturaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura de repeticiones para entrenamiento")

        self.etiquetas = ["sentadilla",
                          "flexion_codo", "flexion", "estiramiento"]
        self.etiqueta_actual = tk.StringVar(value=self.etiquetas[0])
        self.capturando = False
        self.secuencia = []
        self.repeticion = 0

        self.cap = cv2.VideoCapture(0)

        self.crear_interfaz()
        self.mostrar_frame()

    def crear_interfaz(self):
        self.selector = ttk.Combobox(self.root, values=self.etiquetas,
                                     textvariable=self.etiqueta_actual, state="readonly", width=20)
        self.selector.grid(row=0, column=0, padx=10, pady=10)

        self.btn_iniciar = tk.Button(self.root, text="Iniciar Captura",
                                     command=self.iniciar_captura, bg="#4CAF50", fg="white", width=20)
        self.btn_iniciar.grid(row=0, column=1, padx=10)

        self.btn_detener = tk.Button(self.root, text="Detener Captura",
                                     command=self.detener_captura, bg="#f44336", fg="white", width=20)
        self.btn_detener.grid(row=0, column=2, padx=10)

        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=1, column=0, columnspan=3)

        self.status = tk.Label(
            self.root, text="Esperando inicio...", fg="blue")
        self.status.grid(row=2, column=0, columnspan=3, pady=5)

    def mostrar_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if self.capturando:
                    landmarks = results.pose_landmarks.landmark
                    caracteristicas = []
                    for lm in landmarks:
                        caracteristicas.extend(
                            [lm.x, lm.y, lm.z, lm.visibility])
                    caracteristicas.extend(calcular_angulos(landmarks))

                    if len(caracteristicas) == 144:
                        self.secuencia.append(caracteristicas)

                    if len(self.secuencia) >= FRAMES_POR_REPETICION:
                        self.guardar_repeticion()
                        self.capturando = False
                        self.status.config(
                            text="✅ Repetición guardada", fg="green")

            # Mostrar imagen en la GUI
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.mostrar_frame)

    def iniciar_captura(self):
        self.secuencia = []
        self.capturando = True
        self.status.config(
            text=f"Grabando {self.etiqueta_actual.get()}...", fg="red")

    def detener_captura(self):
        self.capturando = False
        self.secuencia = []
        self.status.config(text="⏹️ Captura detenida", fg="orange")

    def guardar_repeticion(self):
        etiqueta = self.etiqueta_actual.get()
        ruta = os.path.join(DATASET_DIR, etiqueta)
        os.makedirs(ruta, exist_ok=True)
        archivo = os.path.join(ruta, f"rep_{len(os.listdir(ruta))+1:03d}.npy")
        np.save(archivo, np.array(self.secuencia))
        print(f"Guardado: {archivo}")

    def cerrar(self):
        self.cap.release()
        self.root.destroy()


# Ejecutar aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = CapturaGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar)
    root.mainloop()
