FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir httpx==0.27.0 psycopg2-binary==2.9.9

COPY run_scheduler.py .

CMD ["python", "run_scheduler.py"]
