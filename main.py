from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AirQ IA Microservice")

# 1. Cargar los modelos empaquetados al iniciar el servidor
try:
    model = joblib.load('airq_universal_model.pkl')
    print("Modelo Universal AirQ (Riesgo) cargado.")
    action_model = joblib.load('airq_action_model.pkl')
    print("Modelo IA para Acciones cargado y listo.")
except Exception as e:
    print(f"Error cargando los modelos: {e}")
    model = None
    action_model = None


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
    if model is None or action_model is None:
        raise HTTPException(status_code=500, detail="Los modelos de IA no están disponibles.")

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

    # 4. Inferencia del Modelo Universal (deduce el riesgo general)
    riesgo = model.predict(features)[0]

    # 5. Inferencia del Modelo de Acciones (100% IA)
    action_id = action_model.predict(features)[0]
    
    co2_val = features['co2'][0]
    pm25_val = features['pm25'][0]
    
    acciones_dict = {
        1: "CRITICAL CONFLICT: Alta toxicidad externa por PM2.5 y asfixia por CO2. Rejillas CERRADAS para aislamiento. Filtro HEPA al 100%. AC en modo COOL. AC en modo DRY. ¡ALERT: Se requiere evacuación del área por imposibilidad de renovación segura de oxígeno!",
        2: "CRITICAL CONFLICT: Alta toxicidad externa por PM2.5 y asfixia por CO2. Rejillas CERRADAS para aislamiento. Filtro HEPA al 100%. AC en modo COOL. ¡ALERT: Se requiere evacuación del área por imposibilidad de renovación segura de oxígeno!",
        3: "CRITICAL CONFLICT: Alta toxicidad externa por PM2.5 y asfixia por CO2. Rejillas CERRADAS para aislamiento. Filtro HEPA al 100%. AC en modo DRY. ¡ALERT: Se requiere evacuación del área por imposibilidad de renovación segura de oxígeno!",
        4: "CRITICAL CONFLICT: Alta toxicidad externa por PM2.5 y asfixia por CO2. Rejillas CERRADAS para aislamiento. Filtro HEPA al 100%. ¡ALERT: Se requiere evacuación del área por imposibilidad de renovación segura de oxígeno!",
        
        5: f"CRITICAL: Contaminación por polvo (PM2.5: {pm25_val}). Rejillas CERRADAS, filtro HEPA al 100%. AC en modo COOL + DRY interno.",
        6: f"CRITICAL: Contaminación por polvo (PM2.5: {pm25_val}). Rejillas CERRADAS, filtro HEPA al 100%. AC en modo COOL interno.",
        7: f"CRITICAL: Contaminación por polvo (PM2.5: {pm25_val}). Rejillas CERRADAS, filtro HEPA al 100%. AC en modo DRY interno.",
        8: f"CRITICAL: Contaminación por polvo (PM2.5: {pm25_val}). Rejillas CERRADAS, filtro HEPA al 100%. Sistemas de climatización en espera.",
        
        9: f"ALERT: Concentración de CO2 alta ({co2_val} ppm). Rejillas ABIERTAS al 100% y Extractores de aire al máximo. AC en modo COOL + DRY activo.",
        10: f"ALERT: Concentración de CO2 alta ({co2_val} ppm). Rejillas ABIERTAS al 100% y Extractores de aire al máximo. AC en modo COOL activo.",
        11: f"ALERT: Concentración de CO2 alta ({co2_val} ppm). Rejillas ABIERTAS al 100% y Extractores de aire al máximo. AC en modo DRY activo.",
        12: f"ALERT: Concentración de CO2 alta ({co2_val} ppm). Rejillas ABIERTAS al 100% y Extractores de aire al máximo. ",
        
        13: "MEDIUM: Aire limpio pero ambiente bochornoso. AC configurado en modo COOL + DRY.",
        14: "MEDIUM: Confort térmico bajo por calor. AC configurado en modo COOL.",
        15: "MEDIUM: Humedad relativa elevada. AC configurado en modo DRY (Deshumidificador).",
        16: "LOW: Calidad del aire óptima y confort térmico adecuado. Todos los actuadores en modo ecológico/espera."
    }
    
    action = acciones_dict.get(action_id, "LOW: Calidad del aire óptima y confort térmico adecuado. Todos los actuadores en modo ecológico/espera.")

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