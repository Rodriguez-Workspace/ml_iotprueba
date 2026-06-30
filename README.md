# AirQ Machine Learning Service (ml_iot)

## Description
This is an independent microservice built with Python and FastAPI. It processes environmental telemetry data received from the AirQ backend to detect anomalies or dangerous trends in air quality.

## Key Features
- **FastAPI Endpoints**: Exposes REST endpoints (e.g., `/predict`) that the Java Backend calls synchronously or asynchronously to evaluate sensor arrays.
- **Predictive Modeling**: Uses Scikit-Learn models to analyze features such as CO2 levels, PM2.5 concentrations, and temperature.
- **Anomaly Detection**: Identifies spikes or critical deterioration in air quality that trigger critical alerts.
- **High Performance**: Designed to run as a stateless microservice, allowing horizontal scaling if telemetry volume grows.

## Tech Stack
- Python 3.9+
- FastAPI & Uvicorn
- Scikit-Learn (scikit-learn)
- Pandas & NumPy
- Joblib (Model serialization)

## How to Run Locally
1. It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install the required data science and web packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI Uvicorn server:
   ```bash
   uvicorn main:app --reload
   ```
4. Access the auto-generated Swagger UI documentation at `http://localhost:8000/docs`.

## Production Deployment
The service is typically deployed on platforms like Render or AWS as a standalone web service, accessed via HTTP by the main Spring Boot backend.
