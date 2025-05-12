from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from pose_detection import detect_pose  # Importar función de detección

app = Flask(__name__)

# Variables globales con bloqueo de hilos
video_capture = None
stream_active = False
lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html', stream_active=stream_active)


@app.route('/start', methods=['POST'])
def start_stream():
    global video_capture, stream_active

    with lock:
        if not stream_active:
            try:
                # Inicializar webcam con resolución optimizada
                video_capture = cv2.VideoCapture(0)

                # Configurar resolución para reducir procesamiento
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                if not video_capture.isOpened():
                    return jsonify({"success": False, "error": "No se pudo acceder a la cámara."})

                stream_active = True
                return jsonify({"success": True})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
    return jsonify({"success": False, "error": "Ya hay una transmisión activa"})


@app.route('/stop', methods=['POST'])
def stop_stream():
    global video_capture, stream_active

    with lock:
        if stream_active and video_capture is not None:
            try:
                video_capture.release()
                video_capture = None
                stream_active = False
                return jsonify({"success": True})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
    return jsonify({"success": False, "error": "No hay transmisión activa"})


@app.route('/video')
def video():
    if not stream_active or (video_capture is not None and not video_capture.isOpened()):
        return "", 204

    def generate_frames():
        with lock:
            if not stream_active:
                return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detectar y procesar postura
            processed_frame = detect_pose(frame)

            # Comprimir imagen con calidad optimizada para reducir uso de CPU
            ret, buffer = cv2.imencode('.jpg', processed_frame, [
                                       int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with lock:
        return jsonify({"active": stream_active})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
