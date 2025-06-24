import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

# ------------------- CONFIGURACIÓN -------------------
VIDEO_DIR = "data/videos/flexion_codo"
SEQUENCE_LENGTH = 60
NUM_COORDS = 6  # x, y de hombro, codo, muñeca (3 puntos * 2)
NUM_ANGLES = 1  # Ángulo del codo
INPUT_DIM = NUM_COORDS + NUM_ANGLES

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils


# ------------------- FUNCIONES AUXILIARES -------------------
def calcular_angulo(a, b, c):
    """Calcula el ángulo entre los vectores AB y BC"""
    ab = np.array([b[0] - a[0], b[1] - a[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    ab_norm = np.linalg.norm(ab)
    bc_norm = np.linalg.norm(bc)

    if ab_norm == 0 or bc_norm == 0:
        return 0.0

    cos_theta = np.dot(ab, bc) / (ab_norm * bc_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)


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
            lm = results.pose_landmarks.landmark

            # Coordenadas x,y del hombro, codo y muñeca derechos
            hombro = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            codo = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            muneca = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Ángulo entre hombro-codo-muñeca
            angulo_codo = calcular_angulo(hombro, codo, muneca)

            # Frame con coordenadas y ángulo
            frame_info = hombro + codo + muneca + [angulo_codo]
            frames.append(frame_info)

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
        y.append(0)  # Clase 0: flexiones de codo

X = np.array(X)
y = to_categorical(y, num_classes=1)  # Ajustar si agregas más clases

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
model.save("modelo_flexiones_codo_lstm.h5")
