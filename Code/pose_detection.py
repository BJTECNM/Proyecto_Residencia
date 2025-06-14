import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# === Cargar modelos específicos ===
modelos = {
    "flexion_codo": load_model("modelo_flexiones_codo_lstm.h5"),
    "flexiones": load_model("modelo_flexiones_lstm.h5"),
    "sentadilla": load_model("modelo_sentadillas_lstm.h5"),
    "estiramiento": load_model("modelo_estiramiento_lstm.h5"),
}

# === MediaPipe ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Parámetros de entrada ===
SEQUENCE_LENGTH = 60
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

            if len(secuencia_actual) == SEQUENCE_LENGTH:
                entrada = np.array(secuencia_actual).reshape(
                    1, SEQUENCE_LENGTH, INPUT_DIM)
                modelo = modelos.get(ejercicio)
                if modelo:
                    prediccion = modelo.predict(entrada, verbose=0)
                    clase_predicha = np.argmax(prediccion)
                    retroalimentacion = interpretar_resultado(
                        clase_predicha, ejercicio)

                    if clase_predicha == 0 and ultima_clase != 0:
                        repeticiones += 1
                    ultima_clase = clase_predicha

    return frame


def interpretar_resultado(clase, ejercicio):
    if ejercicio == "flexion_codo":
        return ["Buena postura", "Codo muy extendido", "Hombro desalineado"][clase] if clase < 3 else "Desconocido"
    elif ejercicio == "flexiones":
        return ["Flexión correcta", "Cuerpo desalineado", "Codos abiertos"][clase] if clase < 3 else "Desconocido"
    elif ejercicio == "sentadilla":
        return ["Postura correcta", "Corrige espalda", "Rodillas mal alineadas"][clase] if clase < 3 else "Desconocido"
    elif ejercicio == "estiramiento":
        return ["Estiramiento correcto", "Brazos mal extendidos", "Torso encorvado"][clase] if clase < 3 else "Desconocido"
    return "Ejercicio desconocido"


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
