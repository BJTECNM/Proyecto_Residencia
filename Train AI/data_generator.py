import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder


# === Aumentos sobre coordenadas ===
def rotar_frame(frame, grados):
    ang = np.radians(grados)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    resultado = []
    for i in range(33):
        x = frame[i * 3]
        y = frame[i * 3 + 1]
        v = frame[i * 3 + 2]
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        resultado.extend([x_rot, y_rot, v])
    resultado.extend(frame[99:])
    return np.array(resultado)


def flip_frame(frame):
    resultado = []
    for i in range(33):
        x = -frame[i * 3]
        y = frame[i * 3 + 1]
        v = frame[i * 3 + 2]
        resultado.extend([x, y, v])
    resultado.extend(frame[99:])
    return np.array(resultado)


def ruido_frame(frame, intensidad=0.01):
    parte_1 = frame[:99] + np.random.normal(0, intensidad, size=99)
    parte_2 = frame[99:]  # ángulos sin ruido
    return np.concatenate([parte_1, parte_2])


def escalar_frame(frame, factor=1.05):
    resultado = []
    for i in range(33):
        x = frame[i * 3] * factor
        y = frame[i * 3 + 1] * factor
        v = frame[i * 3 + 2]
        resultado.extend([x, y, v])
    resultado.extend(frame[99:])
    return np.array(resultado)


def aplicar_aumentos(secuencia):
    nueva = []
    for frame in secuencia:
        if random.random() < 0.3:
            frame = rotar_frame(frame, random.uniform(-5, 5))
        if random.random() < 0.2:
            frame = flip_frame(frame)
        if random.random() < 0.3:
            frame = ruido_frame(frame)
        if random.random() < 0.2:
            frame = escalar_frame(frame, random.uniform(0.95, 1.05))
        nueva.append(frame)
    return np.array(nueva, dtype=np.float32)


# === Generador ===
class PoseSequenceGenerator(Sequence):
    def __init__(self, data_dir, batch_size=16, augment=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment
        self.archivos = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        self._cargar_archivos()
        self.label_encoder.fit(self.labels)
        self.n_classes = len(self.label_encoder.classes_)

    def _cargar_archivos(self):
        for clase in os.listdir(self.data_dir):
            clase_dir = os.path.join(self.data_dir, clase)
            if not os.path.isdir(clase_dir):
                continue
            for archivo in os.listdir(clase_dir):
                if archivo.endswith(".npy"):
                    ruta = os.path.join(clase_dir, archivo)
                    secuencia = np.load(ruta)
                    if secuencia.shape[1] == 105:
                        self.archivos.append(ruta)
                        self.labels.append(clase)
                    else:
                        print(
                            f"❌ Ignorado por shape incorrecto: {ruta} ({secuencia.shape})")

    def __len__(self):
        return len(self.archivos) // self.batch_size

    def __getitem__(self, idx):
        batch_archivos = self.archivos[idx *
                                       self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx *
                                   self.batch_size:(idx + 1) * self.batch_size]

        X, y = [], []
        for path, label in zip(batch_archivos, batch_labels):
            secuencia = np.load(path)
            if self.augment:
                secuencia = aplicar_aumentos(secuencia)
            X.append(secuencia)
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = self.label_encoder.transform(y)
        y = to_categorical(y, num_classes=self.n_classes)
        return X, y

    def on_epoch_end(self):
        # Reorganizar aleatoriamente los datos cada época
        tmp = list(zip(self.archivos, self.labels))
        random.shuffle(tmp)
        self.archivos, self.labels = zip(*tmp)

    def guardar_etiquetas(self, ruta="clases.txt"):
        with open(ruta, "w") as f:
            for clase in self.label_encoder.classes_:
                f.write(clase + "\n")
