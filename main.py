from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load("modelo_ml.pkl")

# Inicializar FastAPI
app = FastAPI()

# Definir un esquema para la entrada de datos
class InputData(BaseModel):
    x: float  # Un número como entrada

@app.post("/predict/")
def predict(data: InputData):
    # Convertir la entrada a un array numpy y hacer la predicción
    X_input = np.array([[data.x]])
    prediction = modelo.predict(X_input)[0]  # Obtener el número predicho
    
    return {"input": data.x, "predicted_value": prediction}

@app.get("/")
def root():
    return {"message": "API de Machine Learning con FastAPI"}