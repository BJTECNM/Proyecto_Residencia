import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose(frame):
    """
    Detecta y marca la postura humana en el frame usando MediaPipe.
    """
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convertir a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar imagen
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Dibujar landmarks y conexiones con OpenCV
            for lm in results.pose_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                x1 = int(results.pose_landmarks.landmark[start_idx].x * frame.shape[1])
                y1 = int(results.pose_landmarks.landmark[start_idx].y * frame.shape[0])
                x2 = int(results.pose_landmarks.landmark[end_idx].x * frame.shape[1])
                y2 = int(results.pose_landmarks.landmark[end_idx].y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return frame
