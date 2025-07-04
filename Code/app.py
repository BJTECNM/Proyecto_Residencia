from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import json
import cv2
import threading
import time
from pose_detection import detect_pose, reiniciar_contador, obtener_repeticiones, obtener_feedback

app = Flask(__name__)

video_capture = None
stream_active = False
lock = threading.Lock()
ejercicio_seleccionado = "flexion_codo"  # Valor inicial


@app.route('/')
def home():
    return redirect(url_for('registro_paciente'))


@app.route('/registro_paciente')
def registro_paciente():
    return render_template('paciente.html')


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


@app.route('/editar/<int:idx>')
def editar_paciente(idx):
    pacientes = cargar_pacientes()
    if idx < len(pacientes):
        return render_template('editar_paciente.html', idx=idx, paciente=pacientes[idx])
    return redirect(url_for('registro_paciente'))


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
    return redirect(url_for('registro_paciente'))


@app.route('/eliminar/<int:idx>')
def eliminar_paciente(idx):
    pacientes = cargar_pacientes()
    if idx < len(pacientes):
        pacientes.pop(idx)
        guardar_todos_pacientes(pacientes)
    return redirect(url_for('registro_paciente'))


@app.route('/buscar_pacientes')
def buscar_pacientes():
    termino = request.args.get('termino', '').strip().lower()
    pacientes = cargar_pacientes()
    coincidencias = []

    for idx, paciente in enumerate(pacientes):
        nombre = paciente.get('nombre', '').lower()
        apellido = paciente.get('apellido', '').lower()
        if termino in nombre or termino in apellido or termino in f"{nombre} {apellido}" or termino in f"{apellido} {nombre}":
            coincidencias.append({
                "index": idx,
                "nombre": paciente['nombre'],
                "apellido": paciente['apellido'],
                "edad": paciente['edad'],
                "complexion": paciente['complexion']
            })

    return jsonify({"pacientes": coincidencias})


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
    global video_capture, stream_active, ejercicio_seleccionado

    data = request.get_json()
    camera_index = data.get('camara_index', 0)
    ejercicio_seleccionado = data.get('ejercicio', 'flexion_codo')

    with lock:
        if not stream_active:
            video_capture = cv2.VideoCapture(camera_index)
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not video_capture.isOpened():
                return jsonify({"success": False, "error": "No se pudo acceder a la cámara."})

            stream_active = True
            reiniciar_contador()

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
        last_time = time.time()

        while True:
            with lock:
                if not stream_active or video_capture is None:
                    break

                ret, frame = video_capture.read()

            if not ret:
                break

            processed_frame = detect_pose(frame, ejercicio_seleccionado)
            ret, buffer = cv2.imencode('.jpg', processed_frame, [
                int(cv2.IMWRITE_JPEG_QUALITY), 85])

            if not ret:
                continue

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # Calcula tiempo transcurrido y ajusta para mantener 30 fps
            current_time = time.time()
            elapsed = current_time - last_time
            last_time = current_time

            # 30 fps (1 / 25) para 25 fps
            time.sleep(max(0, (1 / 25) - elapsed))

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with lock:
        return jsonify({"active": stream_active})


@app.route('/contador')
def contador():
    return jsonify({"repeticiones": obtener_repeticiones()})


@app.route('/feedback')
def feedback():
    return jsonify(obtener_feedback())


# Funciones auxiliares para pacientes
def cargar_pacientes():
    path = os.path.join('data', 'pacientes.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def guardar_todos_pacientes(pacientes):
    os.makedirs('data', exist_ok=True)
    path = os.path.join('data', 'pacientes.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(pacientes, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
