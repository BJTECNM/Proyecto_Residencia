import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# === Cargar modelo LSTM general y clases ===
modelo = load_model("modelo_lstm_ejercicios.h5")
clases = open("clases.txt").read().splitlines()

# === MediaPipe ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Configuración ===
SEQUENCE_LENGTH = 60
INPUT_DIM = 144
UMBRAL_CONF = 0.85

# === Mensajes personalizados por clase incorrecta ===
RETROALIMENTACION_POR_CLASE = {
    "mala_postura": "⚠️ Corrige tu postura",
    "sentadilla_incorrecta": "⚠️ Espalda recta y baja más",
    "flexion_incorrecta": "⚠️ Cuida la alineación de brazos",
    "flexion_codo_incorrecta": "⚠️ Brazos más cerca del torso",
    "estiramiento_incorrecto": "⚠️ Mantén la postura recta"
}

# === Estado global ===
secuencia_actual = []
repeticiones = 0
ultima_clase = ""
retroalimentacion = ""


# === Funciones de procesamiento ===
def calcular_angulos(landmarks):
    def get_angle(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba)
                                   * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    lmk = landmarks.landmark
    return [
        get_angle(lmk[23], lmk[25], lmk[27]),  # pierna izquierda
        get_angle(lmk[24], lmk[26], lmk[28]),  # pierna derecha
        get_angle(lmk[11], lmk[13], lmk[15]),  # brazo izquierdo
        get_angle(lmk[12], lmk[14], lmk[16]),  # brazo derecho
        get_angle(lmk[11], lmk[23], lmk[25]),  # torso izq
        get_angle(lmk[12], lmk[24], lmk[26])   # torso der
    ]


def extract_landmark_data(results):
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark[:33]:
            keypoints.extend([lm.x, lm.y])
    return keypoints if keypoints else None


def detectar_clase(entrada):
    pred = modelo.predict(np.expand_dims(entrada, axis=0), verbose=0)[0]
    idx = np.argmax(pred)
    return clases[idx], pred[idx]


def detect_pose(frame, ejercicio_esperado):
    global secuencia_actual, repeticiones, ultima_clase, retroalimentacion

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2)
        )

        keypoints = extract_landmark_data(results)
        angulos = calcular_angulos(results.pose_landmarks)

        if keypoints and len(angulos) == 6:
            keypoints.extend(angulos)
            keypoints = keypoints[:INPUT_DIM] + \
                [0.0] * (INPUT_DIM - len(keypoints))
            secuencia_actual.append(keypoints)

            if len(secuencia_actual) > SEQUENCE_LENGTH:
                secuencia_actual.pop(0)

            if len(secuencia_actual) < SEQUENCE_LENGTH:
                retroalimentacion = f"Cargando... ({len(secuencia_actual)}/{SEQUENCE_LENGTH})"
            else:
                clase_predicha, confianza = detectar_clase(secuencia_actual)
                if confianza > UMBRAL_CONF:
                    if clase_predicha in RETROALIMENTACION_POR_CLASE:
                        retroalimentacion = RETROALIMENTACION_POR_CLASE[clase_predicha]
                    else:
                        retroalimentacion = f"{clase_predicha} ({confianza:.2f})"

                    if clase_predicha == ejercicio_esperado and ultima_clase != clase_predicha:
                        repeticiones += 1
                    ultima_clase = clase_predicha
                else:
                    retroalimentacion = "Postura no reconocida"

    return frame


def reiniciar_contador():
    global secuencia_actual, repeticiones, ultima_clase, retroalimentacion
    secuencia_actual = []
    repeticiones = 0
    ultima_clase = ""
    retroalimentacion = ""


def obtener_repeticiones():
    return repeticiones


def obtener_feedback():
    return retroalimentacion
