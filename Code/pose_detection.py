import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np


modelo_postura = load_model('modelo_ejercicios_lstm.h5')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_landmark_data(results):
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints if keypoints else None


def detect_pose(frame, ejercicio):
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
            entrada = np.array(keypoints).reshape(1, -1)
            prediccion = modelo_postura.predict(entrada)
            clase_predicha = np.argmax(prediccion)

            # Aquí decides cómo dar retroalimentación y conteo
            feedback = interpretar_resultado(clase_predicha, ejercicio)

            # Mostrar feedback en pantalla
            cv2.putText(frame, feedback, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
