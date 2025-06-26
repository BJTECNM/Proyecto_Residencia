import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_generator import PoseSequenceGenerator  # tu generador con aumentos

# === CONFIGURACIÓN ===
DATA_DIR = "data/dataset"
SEQUENCE_LENGTH = 60
INPUT_DIM = 99  # x, y, visibility por 33 puntos
BATCH_SIZE = 16
EPOCHS = 80

# === Generadores ===
train_gen = PoseSequenceGenerator(
    DATA_DIR, batch_size=BATCH_SIZE, augment=True)
val_gen = PoseSequenceGenerator(DATA_DIR, batch_size=BATCH_SIZE, augment=False)
train_gen.guardar_etiquetas("clases.txt")  # genera clases.txt automáticamente

# === Modelo LSTM ===
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

# === Entrenamiento ===
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# === Guardar modelo
model.save("modelo_ejercicios.h5")
print("✅ Modelo guardado como modelo_ejercicios.h5")
