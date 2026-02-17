FROM python:3.11-slim

WORKDIR /app

# System deps for Prophet
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY outputs /app/outputs

ENV PORT=8000
ENV MODELS_DIR=/app/outputs/models

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "src.api:app"]
