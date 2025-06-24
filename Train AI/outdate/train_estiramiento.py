import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

# ------------------- CONFIGURACIÓN -------------------
VIDEO_DIR = "data/videos/estiramiento"  # Ruta a los videos de estiramiento
SEQUENCE_LENGTH = 60
NUM_ANGLES = 9  # 3 por brazo, 3 por pierna, 3 torso
NUM_COORDS = 33 * 2  # x,y de cada punto
INPUT_DIM = NUM_ANGLES + NUM_COORDS

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils


# ------------------- FUNCIONES AUXILIARES -------------------
def get_angle(a, b, c):
    """Calcula el ángulo entre tres puntos."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)
                                     * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def calcular_angulos(landmarks):
    """Calcula ángulos clave para estiramiento."""
    lms = {i: [lm.x, lm.y] for i, lm in enumerate(landmarks.landmark)}

    # Brazos
    angulo_brazo_izq = get_angle(
        lms[11], lms[13], lms[15])  # hombro-codo-muñeca
    angulo_brazo_der = get_angle(lms[12], lms[14], lms[16])

    # Piernas
    angulo_pierna_izq = get_angle(
        lms[23], lms[25], lms[27])  # cadera-rodilla-tobillo
    angulo_pierna_der = get_angle(lms[24], lms[26], lms[28])

    # Torso: hombro-cadera-rodilla (izquierdo y derecho)
    angulo_torso_izq = get_angle(lms[11], lms[23], lms[25])
    angulo_torso_der = get_angle(lms[12], lms[24], lms[26])

    # Inclinación global del torso (ángulo entre hombros y caderas)
    hombros_vec = np.array(lms[11]) - np.array(lms[12])
    caderas_vec = np.array(lms[23]) - np.array(lms[24])
    inclinacion = get_angle(lms[11], lms[23], lms[24])

    return [
        angulo_brazo_izq, angulo_brazo_der,
        angulo_pierna_izq, angulo_pierna_der,
        angulo_torso_izq, angulo_torso_der,
        inclinacion,
        np.abs(hombros_vec[0]),  # desplazamiento horizontal de hombros
        np.abs(caderas_vec[0]),  # desplazamiento horizontal de caderas
    ]


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
        y.append(0)  # Solo una clase: estiramiento

X = np.array(X)
y = to_categorical(y, num_classes=1)

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
model.save("modelo_estiramiento_lstm.h5")
