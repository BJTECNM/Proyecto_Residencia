from flask import Flask, render_template_string
import cv2
import mediapipe as mp

app = Flask(__name__)

# Plantilla HTML dinámica para mostrar la cámara en tiempo real
HTML_TEMPLATE_VIDEO = '''
<!DOCTYPE html>
<html>
<head><title>Transmisión de Cámara con Flask</title></head>
<body>
    <h1>Cámara Web Procesada por MediaPipe</h1>
    <!-- Imagen que apunta al endpoint /video -->
    <img src="/video" style="width: 100%;" />
</body>
</html>
'''

# Inicialización de MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Captura la cámara web
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE_VIDEO)

@app.route('/video')
def video():
    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convertir a RGB (requerido por MediaPipe)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe Pose
            results = pose.process(image_rgb)

            # Dibujar landmarks si se detecta cuerpo
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)
                )

            # Codificar imagen a formato JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            # Enviar imagen en formato de flujo MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return app.response_class(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # Limpiar recursos al finalizar
    import atexit
    @atexit.register
    def release_resources():
        cap.release()
    
    app.run(debug=True)
