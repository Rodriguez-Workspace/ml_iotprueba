from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AirQ IA Microservice")

# Render usa esto para verificar si el servidor está vivo y sano
@app.get("/")
@app.head("/")
def health_check():
    return {"status": "ok", "message": "ML Service is running"}

# 1. Cargar el cerebro empaquetado al iniciar el servidor
try:
    model = joblib.load('airq_universal_model.pkl')
    print("✅ Modelo Universal AirQ cargado y listo en memoria.")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    model = None


# 2. Definir la estructura de comunicación con Spring Boot (Los 4 sensores)
class SensorData(BaseModel):
    co2: float
    pm25: float
    temp: float
    hum: float


class PredictionRequest(BaseModel):
    sensorId: str
    data: list[SensorData]  # Recibimos la ventana de los últimos minutos


class PredictionResponse(BaseModel):
    riskLevel: str
    aiActionTaken: str


# 3. El Endpoint de Inferencia
@app.post("/predict", response_model=PredictionResponse)
def predict_air_quality(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo de IA no está disponible.")

    if not request.data:
        raise HTTPException(status_code=400, detail="No se enviaron datos de los sensores.")

    # Tomamos el último registro (el más reciente) para la predicción inmediata
    latest_data = request.data[-1]

    # Formateamos exactamente igual a como entrenamos (co2, pm25, temp, hum)
    features = pd.DataFrame([{
        'co2': latest_data.co2,
        'pm25': latest_data.pm25,
        'temp': latest_data.temp,
        'hum': latest_data.hum
    }])

    # 4. Inferencia del Modelo (Random Forest deduce el riesgo general)
    riesgo = model.predict(features)[0]

    co2_alto = features['co2'][0] >= 900
    pm25_alto = features['pm25'][0] >= 35.0
    temp_alta = features['temp'][0] >= 28.0
    hum_alta = features['hum'][0] >= 70.0

    action = ""

    # Implementación estricta de las reglas de control de hardware
    if pm25_alto and co2_alto:
        # Resolución de Conflictos (Casos 13, 14, 15, 16)
        action = "CRITICAL CONFLICT: Alta toxicidad externa por PM2.5 y asfixia por CO2. Rejillas CERRADAS para aislamiento. Filtro HEPA al 100%. "
        if temp_alta: action += "AC en modo COOL. "
        if hum_alta: action += "AC en modo DRY. "
        action += "¡ALERT: Se requiere evacuación del área por imposibilidad de renovación segura de oxígeno!"

    elif pm25_alto:
        # Casos de Aislamiento Puro (Casos 5, 6, 7, 8)
        action = f"CRITICAL: Contaminación por polvo (PM2.5: {features['pm25'][0]}). Rejillas CERRADAS, filtro HEPA al 100%. "
        if temp_alta and hum_alta:
            action += "AC en modo COOL + DRY interno."
        elif temp_alta:
            action += "AC en modo COOL interno."
        elif hum_alta:
            action += "AC en modo DRY interno."
        else:
            action += "Sistemas de climatización en espera."

    elif co2_alto:
        # Casos de Ventilación Pura (Casos 9, 10, 11, 12)
        action = f"ALERT: Concentración de CO2 alta ({features['co2'][0]} ppm). Rejillas ABIERTAS al 100% y Extractores de aire al máximo. "
        if temp_alta and hum_alta:
            action += "AC en modo COOL + DRY activo."
        elif temp_alta:
            action += "AC en modo COOL activo."
        elif hum_alta:
            action += "AC en modo DRY activo."

    else:
        # Casos de Confort Térmico/Humedad (Casos 1, 2, 3, 4)
        if temp_alta and hum_alta:
            action = "MEDIUM: Aire limpio pero ambiente bochornoso. AC configurado en modo COOL + DRY."
        elif temp_alta:
            action = "MEDIUM: Confort térmico bajo por calor. AC configurado en modo COOL."
        elif hum_alta:
            action = "MEDIUM: Humedad relativa elevada. AC configurado en modo DRY (Deshumidificador)."
        else:
            action = "LOW: Calidad del aire óptima y confort térmico adecuado. Todos los actuadores en modo ecológico/espera."

    return PredictionResponse(
        riskLevel=riesgo,
        aiActionTaken=action
    )


if __name__ == "__main__":
    import uvicorn
    import os

    # Render sets the PORT environment variable. Fallback to 5000 for local.
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)