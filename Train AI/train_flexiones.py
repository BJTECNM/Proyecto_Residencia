import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

# ------------------- CONFIGURACIÓN -------------------
VIDEO_DIR = "data/videos/flexiones"  # Carpeta con videos de flexiones
SEQUENCE_LENGTH = 60
NUM_ANGLES = 6
NUM_COORDS = 33 * 2  # x, y por punto
INPUT_DIM = NUM_ANGLES + NUM_COORDS

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# ------------------- FUNCIONES AUXILIARES -------------------
def calcular_angulo(a, b, c):
    """Calcula el ángulo entre tres puntos"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    cb = c - b

    radians = np.arccos(
        np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    )
    return np.degrees(radians)

def calcular_angulos(landmarks):
    """Devuelve 6 ángulos relevantes para flexiones"""
    # Puntos claves por índice (ver doc de MediaPipe Pose)
    puntos = landmarks.landmark
    angulos = []

    # Ángulos: hombro-codo-muñeca (ambos lados), cadera-hombro-codo, rodilla-cadera-hombro
    angulos.append(calcular_angulo(
        [puntos[12].x, puntos[12].y],  # right_shoulder
        [puntos[14].x, puntos[14].y],  # right_elbow
        [puntos[16].x, puntos[16].y]   # right_wrist
    ))
    angulos.append(calcular_angulo(
        [puntos[11].x, puntos[11].y],  # left_shoulder
        [puntos[13].x, puntos[13].y],  # left_elbow
        [puntos[15].x, puntos[15].y]   # left_wrist
    ))
    angulos.append(calcular_angulo(
        [puntos[24].x, puntos[24].y],  # right_hip
        [puntos[12].x, puntos[12].y],  # right_shoulder
        [puntos[14].x, puntos[14].y]   # right_elbow
    ))
    angulos.append(calcular_angulo(
        [puntos[23].x, puntos[23].y],  # left_hip
        [puntos[11].x, puntos[11].y],  # left_shoulder
        [puntos[13].x, puntos[13].y]   # left_elbow
    ))
    angulos.append(calcular_angulo(
        [puntos[26].x, puntos[26].y],  # right_knee
        [puntos[24].x, puntos[24].y],  # right_hip
        [puntos[12].x, puntos[12].y]   # right_shoulder
    ))
    angulos.append(calcular_angulo(
        [puntos[25].x, puntos[25].y],  # left_knee
        [puntos[23].x, puntos[23].y],  # left_hip
        [puntos[11].x, puntos[11].y]   # left_shoulder
    ))

    return angulos

def procesar_video(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmark_frame = []
            for lm in results.pose_landmarks.landmark:
                landmark_frame.extend([lm.x, lm.y])
            if len(landmark_frame) == NUM_COORDS:
                angulos = calcular_angulos(results.pose_landmarks)
                frames.append(landmark_frame + angulos)

    cap.release()

    # Padding o truncado
    if len(frames) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(frames), INPUT_DIM))
        frames = np.vstack((frames, pad))
    else:
        frames = np.array(frames[:SEQUENCE_LENGTH])

    return frames

# ------------------- PROCESAR DATOS -------------------
X = []
y = []

for archivo in os.listdir(VIDEO_DIR):
    if archivo.endswith(".mp4"):
        ruta = os.path.join(VIDEO_DIR, archivo)
        secuencia = procesar_video(ruta)
        X.append(secuencia)
        y.append(0)  # Clase 0: flexiones

X = np.array(X)
y = to_categorical(y, num_classes=1)  # Ajustable si agregas más ejercicios

# ------------------- MODELO -------------------
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(SEQUENCE_LENGTH, INPUT_DIM)))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ------------------- ENTRENAMIENTO -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=30, batch_size=8,
          validation_data=(X_test, y_test))

# ------------------- GUARDADO -------------------
model.save("modelo_flexiones_lstm.h5")
