from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
from pose_detection import detect_pose

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

    data = request.get_json()
    camera_index = data.get('camara_index', 0)

    with lock:
        if not stream_active:
            video_capture = cv2.VideoCapture(camera_index)
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not video_capture.isOpened():
                return jsonify({"success": False, "error": "No se pudo acceder a la cámara."})

            stream_active = True
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Ya hay una transmisión activa"})


@app.route('/stop', methods=['POST'])
def stop_stream():
    global video_capture, stream_active

    with lock:
        if stream_active:
            if video_capture:
                video_capture.release()
                video_capture = None
            stream_active = False
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "No hay transmisión activa"})


@app.route('/video')
def video():
    def generate_frames():
        global video_capture

        while True:
            with lock:
                if not stream_active or video_capture is None:
                    break

                ret, frame = video_capture.read()

            if not ret:
                break

            proccessed_frame = detect_pose(frame)
            ret, buffer = cv2.imencode('.jpg', proccessed_frame, [
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
