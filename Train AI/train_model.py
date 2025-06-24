import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# === CONFIGURACIÓN ===
DATASET_DIR = "data/dataset"
FRAMES = 60
FEATURES = 144
OUTPUT_MODEL = "data/modelo_ejercicios.h5"

# === Cargar datos ===
X = []
y = []

for etiqueta in os.listdir(DATASET_DIR):
    ruta_etiqueta = os.path.join(DATASET_DIR, etiqueta)
    if not os.path.isdir(ruta_etiqueta):
        continue
    for archivo in os.listdir(ruta_etiqueta):
        if archivo.endswith(".npy"):
            data = np.load(os.path.join(ruta_etiqueta, archivo))
            if data.shape == (FRAMES, FEATURES):
                X.append(data)
                y.append(etiqueta)

X = np.array(X)  # (num_samples, 60, 144)
y = np.array(y)

print(f"Datos cargados: {X.shape}, Etiquetas: {y.shape}")

# === Preprocesar etiquetas ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Guardar clases
with open("clases.txt", "w") as f:
    for clase in le.classes_:
        f.write(clase + "\n")

# === Dividir datos ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y)

# === Definir modelo LSTM ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(FRAMES, FEATURES)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Entrenar modelo ===
checkpoint = ModelCheckpoint(
    OUTPUT_MODEL, monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit(X_train, y_train, validation_data=(
    X_val, y_val), epochs=40, batch_size=16, callbacks=[checkpoint])

print(f"✅ Modelo guardado como: {OUTPUT_MODEL}")
