FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV and pyttsx3/gTTS
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY server-requirements.txt .
RUN pip install --no-cache-dir -r server-requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY server.py server_pipeline.py ./

EXPOSE 8000

CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000"]
