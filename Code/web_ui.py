from flask import Flask, render_template_string, Response, jsonify
import cv2
import threading

app = Flask(__name__)

# Variables globales con bloqueo de hilos
video_capture = None
stream_active = False
lock = threading.Lock()

HTML_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Transmisión Webcam</title>
    <style>
        body { font-family: Arial; text-align: center; padding-top: 50px; }
        #video-feed {
            width: 640px;
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <h1>Transmisión en Vivo con Flask</h1>
    <!-- Botones de control -->
    <button id="startBtn" onclick="startStream()" {% if stream_active %}disabled{% endif %}>Iniciar</button>
    <button id="stopBtn" onclick="stopStream()" disabled>Detener</button>

    <br><br>
    <img id="video-feed" src="" alt="Video feed">

    <script>
        let isStreaming = false;

        // Inicia transmisión
        function startStream() {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;

            fetch('/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Actualiza la imagen para mostrar el stream
                        document.getElementById('video-feed').src = 
                            `/video?${new Date().getTime()}`;
                        isStreaming = true;
                    } else {
                        alert(`Error: ${data.error}`);
                        document.getElementById('startBtn').disabled = false;
                    }
                })
                .catch(err => {
                    console.error(err);
                    alert('Error en el servidor');
                    document.getElementById('startBtn').disabled = false;
                });
        }

        // Detiene transmisión
        function stopStream() {
            if (!isStreaming) return;

            document.getElementById('stopBtn').disabled = true;
            document.getElementById('startBtn').disabled = false;

            fetch('/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('video-feed').src = '';
                        isStreaming = false;
                    } else {
                        alert(`Error: ${data.error}`);
                        document.getElementById('stopBtn').disabled = false;
                    }
                })
                .catch(err => {
                    console.error(err);
                    alert('Error en el servidor');
                    document.getElementById('stopBtn').disabled = false;
                });
        }

        // Estado inicial al cargar la página
        window.addEventListener('load', () => {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.active) {
                        document.getElementById('video-feed').src = `/video?${new Date().getTime()}`;
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                    }
                });
        });
    </script>
</body>
</html>
"""

# Función para generar el flujo de video
def generate_frames():
    with lock:
        if not stream_active or not video_capture.isOpened():
            return

    while True:
        success, frame = video_capture.read()
        if not success:
            break
            
        # Codificar frame a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Ruta principal
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, stream_active=stream_active)

# Iniciar transmisión
@app.route('/start', methods=['POST'])
def start_stream():
    global video_capture, stream_active
    
    with lock:
        if not stream_active:
            try:
                # Inicializar webcam (0 para cámara integrada)
                video_capture = cv2.VideoCapture(0)
                
                # Verificar si la cámara está funcionando
                if not video_capture.isOpened():
                    return jsonify({"success": False, "error": "No se pudo acceder a la cámara."})
                    
                stream_active = True
                return jsonify({"success": True})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
    
    return jsonify({"success": False, "error": "Ya hay una transmisión activa"})

# Detener transmisión
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

# Flujo de video en tiempo real (MJPEG)
@app.route('/video')
def video():
    if not stream_active or not video_capture.isOpened():
        return "Cámara no disponible", 404
        
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Estado actual
@app.route('/status')
def status():
    global stream_active
    with lock:
        return jsonify({"active": stream_active})

if __name__ == '__main__':
    app.run(debug=True)
