import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from data_generator import PoseSequenceGenerator  # tu generador personalizado

# === CONFIGURACIÓN GENERAL ===
DATA_DIR = "data/dataset"
SEQUENCE_LENGTH = 60
INPUT_DIM = 105  # 33 puntos * (x, y, visibilidad)
BATCH_SIZE = 16
EPOCHS = 80

# === Generadores de datos ===
train_gen = PoseSequenceGenerator(
    DATA_DIR, batch_size=BATCH_SIZE, augment=True)
val_gen = PoseSequenceGenerator(DATA_DIR, batch_size=BATCH_SIZE, augment=False)
train_gen.guardar_etiquetas("clases.txt")

# === Definición del modelo LSTM ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, INPUT_DIM)),
    Dropout(0.4),
    LSTM(64),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(train_gen.n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# === EarlyStopping ===
early_stop = EarlyStopping(
    monitor='val_loss',       # Monitorea la pérdida de validación
    patience=10,              # Si no mejora en 10 épocas, se detiene
    restore_best_weights=True  # Restaura los pesos de la mejor época
)

# === Entrenamiento ===
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# === Guardado del modelo en formato moderno
model.save("modelo2_ejercicios.keras")
print("✅ Modelo guardado como modelo2_ejercicios.keras")
