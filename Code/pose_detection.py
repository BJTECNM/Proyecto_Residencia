import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# === Cargar modelos específicos ===
modelos = {
    "flexion_codo": load_model("modelo_flexion_codo.h5"),
    # "flexiones": load_model("modelo_flexiones_lstm.h5"),
    "sentadilla": load_model("modelo_sentadillas.h5"),
    "estiramiento": load_model("modelo_estiramiento.h5"),
}

# === Diccionario de etiquetas por modelo ===
ETIQUETAS_MODELOS = {
    "flexion_codo": ["De pie", "Flexión de codo", "No se puede detectar bien", "Coloque al paciente dentro del área de captura"],
    "flexiones": ["De pie", "Flexiones", "No se puede detectar bien", "Coloque al paciente dentro del área de captura"],
    "sentadilla": ["De pie", "Sentadilla", "No se puede detectar bien", "Coloque al paciente dentro del área de captura"],
    "estiramiento": ["De pie", "Estiramiento", "No se puede detectar bien", "Coloque al paciente dentro del área de captura"]
}

# === MediaPipe ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Parámetros de entrada ===
SEQUENCE_LENGTH = 205
INPUT_DIM = 72

# === Estado global ===
secuencia_actual = []
repeticiones = 0
ultima_clase = None
retroalimentacion = ""


# === Función para calcular ángulos básicos ===
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
        get_angle(lmk[12], lmk[24], lmk[26]),  # torso der
    ]


def extract_landmark_data(results):
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark[:33]:
            keypoints.extend([lm.x, lm.y])
    return keypoints if keypoints else None


def detect_pose(frame, ejercicio):
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

            if len(secuencia_actual) == SEQUENCE_LENGTH:
                vector = np.array(secuencia_actual).flatten()
                if vector.shape[0] < 14739:
                    vector = np.pad(vector, (0, 14739 - vector.shape[0]))
                entrada = vector[:14739].reshape(1, -1)

                modelo = modelos.get(ejercicio)
                if modelo:
                    prediccion = modelo.predict(entrada, verbose=0)
                    clase_predicha = np.argmax(prediccion)
                    retroalimentacion = interpretar_resultado(
                        clase_predicha, ejercicio)

                    # Reglas de conteo: de 'De pie' (0) a 'Ejercicio' (1)
                    if ultima_clase == 0 and clase_predicha == 1:
                        repeticiones += 1
                    ultima_clase = clase_predicha
                else:
                    retroalimentacion = "Modelo no disponible"

    return frame


def interpretar_resultado(clase, ejercicio):
    etiquetas = ETIQUETAS_MODELOS.get(ejercicio, [])
    return etiquetas[clase] if clase < len(etiquetas) else "Desconocido"


def reiniciar_contador():
    global secuencia_actual, repeticiones, ultima_clase, retroalimentacion
    secuencia_actual = []
    repeticiones = 0
    ultima_clase = None
    retroalimentacion = ""


def obtener_repeticiones():
    return repeticiones


def obtener_feedback():
    return retroalimentacion
