import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os
import numpy as np
import cv2
import mediapipe as mp

# === MediaPipe ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


# === Visualización manual ===
def visualizar_npy(filepath, clase):
    try:
        secuencia = np.load(filepath)
    except Exception as e:
        messagebox.showerror("Error al abrir archivo",
                             f"No se pudo abrir el archivo:\n{e}")
        return

    # Validar forma de datos
    if len(secuencia.shape) != 2 or secuencia.shape[1] < 66:
        messagebox.showerror(
            "Archivo inválido",
            f"El archivo no tiene la forma esperada.\n"
            f"Se esperaban al menos 66 columnas (33 puntos x,y) pero se recibieron {secuencia.shape[1]}."
        )
        return

    print(
        f"🔍 Visualizando: {filepath} | Shape: {secuencia.shape} | Clase: {clase}")
    frame_idx = 0

    while True:
        frame_data = secuencia[frame_idx]
        puntos = frame_data[:66]  # 33 puntos x,y
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        coords = []

        for i in range(0, len(puntos), 2):
            x = int(puntos[i] * 640)
            y = int(puntos[i + 1] * 480)
            coords.append((x, y))
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        for connection in POSE_CONNECTIONS:
            i1, i2 = connection
            if i1 < len(coords) and i2 < len(coords):
                cv2.line(img, coords[i1], coords[i2], (0, 255, 0), 2)

        cv2.putText(img, f"Clase: {clase}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
        cv2.putText(img, f"Frame: {frame_idx+1}/{len(secuencia)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)

        cv2.imshow("Secuencia NPY (Usa ← → y Q para salir)", img)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == 81 or key == ord('a'):  # izquierda
            frame_idx = max(0, frame_idx - 1)
        elif key == 83 or key == ord('d'):  # derecha
            frame_idx = min(len(secuencia) - 1, frame_idx + 1)

    cv2.destroyAllWindows()


# === Interfaz gráfica con árbol de directorios ===
class VisualizadorApp:
    def __init__(self, root, carpeta_base="data/dataset"):
        self.root = root
        self.root.title("Visualizador de Secuencias NPY")
        self.root.geometry("600x420")
        self.carpeta_base = carpeta_base

        self.tree = ttk.Treeview(root)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        self.tree.heading('#0', text='Archivos .npy en dataset')

        self.btn_ver = tk.Button(root, text="Visualizar archivo seleccionado", font=("Arial", 12),
                                 command=self.visualizar_seleccionado)
        self.btn_ver.pack(pady=5)

        self.archivos = {}
        self.poblar_arbol()

    def poblar_arbol(self):
        for carpeta, subdirs, archivos in os.walk(self.carpeta_base):
            parent = ''
            ruta_relativa = os.path.relpath(carpeta, self.carpeta_base)
            if ruta_relativa == ".":
                nodo = ''
            else:
                secciones = ruta_relativa.split(os.sep)
                padre = ''
                for seccion in secciones:
                    padre_id = self.archivos.get(padre + "/" + seccion)
                    if padre_id:
                        padre = padre_id
                    else:
                        nodo_id = self.tree.insert(
                            padre, "end", text=seccion, open=False)
                        self.archivos[os.path.join(padre, seccion)] = nodo_id
                        padre = nodo_id
                nodo = padre

            for archivo in archivos:
                if archivo.endswith(".npy"):
                    path_completo = os.path.join(carpeta, archivo)
                    archivo_id = self.tree.insert(
                        nodo, "end", text=archivo, values=[path_completo])
                    self.archivos[path_completo] = archivo_id

    def visualizar_seleccionado(self):
        item = self.tree.focus()
        if not item:
            messagebox.showwarning("Aviso", "Selecciona un archivo .npy")
            return

        texto = self.tree.item(item, "text")
        padre = self.tree.parent(item)

        # Reconstruir ruta
        ruta = [texto]
        while padre:
            ruta.insert(0, self.tree.item(padre, "text"))
            padre = self.tree.parent(padre)

        filepath = os.path.join(self.carpeta_base, *ruta)
        if not filepath.endswith(".npy"):
            messagebox.showwarning(
                "Archivo inválido", "Selecciona un archivo .npy válido")
            return

        # Clase = nombre de la carpeta que contiene el archivo
        clase = ruta[-2] if len(ruta) >= 2 else "desconocido"
        visualizar_npy(filepath, clase)


# === Ejecutar la aplicación ===
if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizadorApp(root)
    root.mainloop()
