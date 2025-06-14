import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

# ------------------- CONFIGURACIÓN -------------------
VIDEO_DIR = "data/videos/sentadillas"  # Carpeta con .mp4 de sentadillas
SEQUENCE_LENGTH = 60
NUM_ANGLES = 6
NUM_COORDS = 33 * 2
INPUT_DIM = NUM_ANGLES + NUM_COORDS

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# ------------------- FUNCIONES AUXILIARES -------------------
def calcular_angulos(landmarks):
    # Placeholder para 6 ángulos – personaliza según tu ejercicio
    return [0] * 6

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
        y.append(0)  # Solo una clase: sentadillas

X = np.array(X)
y = to_categorical(y, num_classes=1)  # Solo 1 clase por ahora

# ------------------- MODELO -------------------
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(SEQUENCE_LENGTH, INPUT_DIM)))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))  # Una sola clase

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ------------------- ENTRENAMIENTO -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=30, batch_size=8,
          validation_data=(X_test, y_test))

# ------------------- GUARDADO -------------------
model.save("modelo_sentadillas_lstm.h5")
