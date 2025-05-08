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
                # Inicializar webcam (0 para cámara integrada)
                video_capture = cv2.VideoCapture(0)
                
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
                # Liberar recursos de la cámara
                video_capture.release()
                video_capture = None
                stream_active = False
                return jsonify({"success": True})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
    
    return jsonify({"success": False, "error": "No hay transmisión activa"})

@app.route('/video')
def video():
    if not stream_active or not video_capture.isOpened():
        return "Cámara no disponible", 404

    def generate_frames():
        with lock:
            if not stream_active:
                return

        while True:
            success, frame = video_capture.read()
            if not success:
                break
            
            # Detectar postura en el frame
            processed_frame = detect_pose(frame)  
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global stream_active
    with lock:
        return jsonify({"active": stream_active})

if __name__ == '__main__':
    app.run(debug=True)
