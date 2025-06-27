import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from tensorflow.keras.models import load_model

# === Cargar modelo LSTM multiclase y etiquetas ===
modelo = load_model("modelo_ejercicios.keras")
clases = open("clases.txt").read().splitlines()

# === MediaPipe ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Configuración ===
SEQUENCE_LENGTH = 60
INPUT_DIM = 105
UMBRAL_CONF = 0.85
pygame.mixer.init()

# Carga sonidos
sonido_inicio = pygame.mixer.Sound("data/audio/start.mp3")
sonido_fin = pygame.mixer.Sound("data/audio/end.mp3")

# === Mensajes personalizados por clase incorrecta ===
RETROALIMENTACION_POR_CLASE = {
    "mala_postura": "⚠️ Corrige tu postura",
    "sentadilla": "Sentadilla realizada",
    "flexion": "Flexión realizada",
    "flexion_codo": "Flexion de codo realizada",
    "estiramiento": "Estiramiento realizado",
    "de_pie": "No se está realizando ningún ejercicio"
}

# === Variables globales ===
secuencia_actual = []
repeticiones = 0
ultima_clase = None
retroalimentacion = ""
esperando_siguiente = False
capturando = False


# --- Funciones auxiliares ---
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


# --- Función principal para procesar cada frame ---
def detect_pose(frame, ejercicio_esperado):
    global secuencia_actual, repeticiones, ultima_clase, retroalimentacion
    global capturando, ultima_captura_time, esperando_siguiente

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb_frame)

    now = time.time()

    if not capturando and not esperando_siguiente:
        # Reproducir sonido de inicio y comenzar captura
        sonido_inicio.play()
        capturando = True
        secuencia_actual = []
        retroalimentacion = "¡Comienza ahora!"

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

        if capturando and keypoints and len(angulos) == 6:
            caracteristicas = keypoints + angulos
            caracteristicas = caracteristicas[:INPUT_DIM] + \
                [0.0] * (INPUT_DIM - len(caracteristicas))
            secuencia_actual.append(caracteristicas)

            retroalimentacion = f"Grabando... ({len(secuencia_actual)}/{SEQUENCE_LENGTH})"

            if len(secuencia_actual) == SEQUENCE_LENGTH:
                # Fin de captura
                capturando = False
                esperando_siguiente = True
                ultima_captura_time = now

                sonido_fin.play()

                entrada = np.array(secuencia_actual)
                clase_predicha, confianza = detectar_clase(entrada)

                if confianza > UMBRAL_CONF:
                    retroalimentacion = RETROALIMENTACION_POR_CLASE.get(
                        clase_predicha, f"{clase_predicha} ({confianza:.2f})")
                    if clase_predicha == ejercicio_esperado and ultima_clase != clase_predicha:
                        repeticiones += 1
                    ultima_clase = clase_predicha
                else:
                    retroalimentacion = "Postura no reconocida"

    if esperando_siguiente and now - ultima_captura_time >= 4.0:
        esperando_siguiente = False
        retroalimentacion = "Preparándote..."

    return frame


# --- Funciones para control externo ---
def reiniciar_contador():
    global secuencia_actual, repeticiones, ultima_clase, retroalimentacion
    secuencia_actual = []
    repeticiones = 0
    ultima_clase = None
    retroalimentacion = ""


def obtener_repeticiones():
    return repeticiones


def obtener_feedback():
    return {
        "mensaje": retroalimentacion,
        "esperando": esperando_siguiente
    }
