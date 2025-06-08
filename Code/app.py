from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import json
import cv2
import threading
from pose_detection import detect_pose

app = Flask(__name__)

# Variables globales
video_capture = None
stream_active = False
lock = threading.Lock()


# Ruta inicial
@app.route('/')
def home():
    return redirect(url_for('registro_paciente'))


@app.route('/registro_paciente')
def registro_paciente():
    pacientes = cargar_pacientes()
    return render_template('paciente.html', pacientes=pacientes)


@app.route('/guardar_paciente', methods=['POST'])
def guardar_paciente():
    datos = {
        "nombre": request.form['nombre'],
        "apellido": request.form['apellido'],
        "edad": request.form['edad'],
        "complexion": request.form['complexion']
    }

    pacientes = cargar_pacientes()
    pacientes.append(datos)
    guardar_todos_pacientes(pacientes)

    return redirect(url_for('pantalla_captura', nombre=datos['nombre'], apellido=datos['apellido'], edad=datos['edad']))


@app.route('/seleccionar_paciente', methods=['POST'])
def seleccionar_paciente():
    nombre = request.form.get('nombre_busqueda')
    pacientes = cargar_pacientes()
    for paciente in pacientes:
        if nombre.lower() in paciente['nombre'].lower():
            return redirect(url_for('pantalla_captura', nombre=paciente['nombre'], apellido=paciente['apellido'], edad=paciente['edad']))
    return redirect(url_for('registro_paciente'))


@app.route('/pacientes')
def lista_pacientes():
    pacientes = cargar_pacientes()
    return render_template('lista_pacientes.html', pacientes=pacientes)


@app.route('/editar/<int:idx>')
def editar_paciente(idx):
    pacientes = cargar_pacientes()
    if idx < len(pacientes):
        return render_template('editar_paciente.html', idx=idx, paciente=pacientes[idx])
    return redirect(url_for('lista_pacientes'))


@app.route('/actualizar/<int:idx>', methods=['POST'])
def actualizar_paciente(idx):
    pacientes = cargar_pacientes()
    if idx < len(pacientes):
        pacientes[idx] = {
            "nombre": request.form['nombre'],
            "apellido": request.form['apellido'],
            "edad": request.form['edad'],
            "complexion": request.form['complexion']
        }
        guardar_todos_pacientes(pacientes)
    return redirect(url_for('lista_pacientes'))


@app.route('/eliminar/<int:idx>')
def eliminar_paciente(idx):
    pacientes = cargar_pacientes()
    if idx < len(pacientes):
        pacientes.pop(idx)
        guardar_todos_pacientes(pacientes)
    return redirect(url_for('lista_pacientes'))


@app.route('/captura')
def pantalla_captura():
    return render_template('index.html', stream_active=stream_active,
                           paciente={
                               "nombre": request.args.get('nombre'),
                               "apellido": request.args.get('apellido'),
                               "edad": request.args.get('edad')
                           })


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


# Utilidades
def cargar_pacientes():
    if os.path.exists('pacientes.json'):
        with open('pacientes.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def guardar_todos_pacientes(pacientes):
    with open('pacientes.json', 'w', encoding='utf-8') as f:
        json.dump(pacientes, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
