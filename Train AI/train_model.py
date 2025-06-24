import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# === CONFIGURACIÓN ===
DATA_DIR = "data/dataset"
SEQUENCE_LENGTH = 150
INPUT_DIM = 144

# === Carga de datos ===
X, y = [], []

for clase in os.listdir(DATA_DIR):
    clase_dir = os.path.join(DATA_DIR, clase)
    if not os.path.isdir(clase_dir):
        continue
    for archivo in os.listdir(clase_dir):
        if archivo.endswith(".npy"):
            ruta = os.path.join(clase_dir, archivo)
            datos = np.load(ruta)
            if datos.shape == (SEQUENCE_LENGTH, INPUT_DIM):
                X.append(datos)
                y.append(clase)

X = np.array(X)
y = np.array(y)

# === Codificar etiquetas ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Guardar las etiquetas
with open("clases.txt", "w") as f:
    for label in le.classes_:
        f.write(label + "\n")

# === División en entrenamiento y validación ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# === Modelo LSTM ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, INPUT_DIM)),
    Dropout(0.4),
    LSTM(64),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# === Entrenamiento ===
model.fit(X_train, y_train, epochs=80,
          validation_data=(X_val, y_val), batch_size=16)

# === Guardar modelo
model.save("modelo_ejercicios.h5")
print("✅ Modelo guardado como modelo_lstm_ejercicios.h5")
