FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/trading/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trading engine + training utilities it imports
COPY backend/trading/ ./
COPY backend/training/model_store.py ./

CMD ["python", "engine.py", "--loop"]
