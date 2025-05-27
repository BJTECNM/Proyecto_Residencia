import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Cargar datos
df = pd.read_csv("modelo_ejercicios.csv")
X = df.drop('label', axis=1)
y = df['label']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Guardar modelo
joblib.dump(clf, "modelo_ejercicios.pkl")
print("[INFO] Modelo guardado como modelo_ejercicios.pkl")
