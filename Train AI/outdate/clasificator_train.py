import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

# ---------- Configuración ----------
DATA_DIR = "data"
SEQUENCE_LENGTH = 60  # Máximo de frames por repetición
NUM_ANGLES = 6
NUM_COORDS = 33 * 2  # 33 puntos x, y
INPUT_DIM = NUM_ANGLES + NUM_COORDS

# Mapear etiquetas
labels = sorted(os.listdir(DATA_DIR))  # ['estiramiento', 'flexion_codo', ...]
label_map = {label: i for i, label in enumerate(labels)}

# ---------- Cargar los datos ----------
X = []
y = []

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for archivo in os.listdir(folder):
        if archivo.endswith(".npy"):
            datos = np.load(os.path.join(folder, archivo))

            # Padding o truncado para igualar longitudes
            if datos.shape[0] < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - datos.shape[0], INPUT_DIM))
                datos = np.vstack((datos, pad))
            else:
                datos = datos[:SEQUENCE_LENGTH]

            X.append(datos)
            y.append(label_map[label])

X = np.array(X)
y = to_categorical(y)

# ---------- División de datos ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- Modelo ----------
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(SEQUENCE_LENGTH, INPUT_DIM)))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ---------- Entrenamiento ----------
model.fit(X_train, y_train, epochs=30, batch_size=16,
          validation_data=(X_test, y_test))

# ---------- Guardar modelo ----------
model.save("modelo_ejercicios_lstm.h5")
