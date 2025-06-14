import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np


modelo_postura = load_model('modelo_ejercicios_lstm.h5')
modelo_sentadilla = load_model('modelo_sentadillas_lstm.h5')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables globales
secuencia_actual = []
SEQUENCE_LENGTH = 60
INPUT_DIM = 72
repeticiones = 0
ultima_clase = None
retroalimentacion = ""


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
        get_angle(lmk[23], lmk[25], lmk[27]),
        get_angle(lmk[24], lmk[26], lmk[28]),
        get_angle(lmk[11], lmk[23], lmk[25]),
        get_angle(lmk[12], lmk[24], lmk[26]),
        get_angle(lmk[13], lmk[11], lmk[23]),
        get_angle(lmk[14], lmk[12], lmk[24]),
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
        if keypoints:
            while len(keypoints) < INPUT_DIM:
                keypoints.append(0.0)  # padding si faltan puntos

            secuencia_actual.append(keypoints)
            if len(secuencia_actual) > SEQUENCE_LENGTH:
                secuencia_actual.pop(0)

            if len(secuencia_actual) == SEQUENCE_LENGTH:
                entrada = np.array(secuencia_actual).reshape(
                    1, SEQUENCE_LENGTH, INPUT_DIM)
                prediccion = modelo_postura.predict(entrada, verbose=0)
                clase_predicha = np.argmax(prediccion)

                retroalimentacion = interpretar_resultado(
                    clase_predicha, ejercicio)

                if clase_predicha == 0 and ultima_clase != 0:
                    repeticiones += 1
                ultima_clase = clase_predicha

                # Mostrar retroalimentación
                cv2.putText(frame, f"{retroalimentacion}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"Repeticiones: {repeticiones}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return frame


# def detect_pose(frame):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2)
        )
    return frame

#


# Falta por definirlo y asegurarnos que funcione
def interpretar_resultado(clase, ejercicio):
    if ejercicio == "flexion_codo":
        if clase == 0:
            return "Buena postura"
        elif clase == 1:
            return "Codo muy extendido"
        elif clase == 2:
            return "Hombro desalineado"
    elif ejercicio == "sentadilla":
        if clase == 0:
            return "Postura correcta"
        else:
            return "Corrige la espalda"
    return "Analizando..."


def reiniciar_contador():
    global secuencia_actual, repeticiones, ultima_clase, retroalimentacion
    secuencia_actual = []
    repeticiones = 0
    ultima_clase = None
    retroalimentacion = ""


def obtener_repeticiones():
    global repeticiones
    return repeticiones


def obtener_feedback():
    global retroalimentacion
    return retroalimentacion
