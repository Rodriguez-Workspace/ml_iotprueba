FROM python:3.11-slim

# Evitar la creación de archivos .pyc para ahorrar espacio y deshabilitar buffering de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar solo requirements primero para aprovechar la caché de Docker en las dependencias
COPY requirements.txt .

# Instalar las dependencias de Python (deshabilitando caché de pip para mantener la imagen ligera)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y el modelo pre-entrenado (.pkl)
COPY . .

# Exponer el puerto por el que correrá la app (Render usará la variable de entorno PORT)
EXPOSE 5000

# Comando para arrancar el servidor web
CMD ["python", "main.py"]
