FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    httpx==0.27.0 \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.1

COPY run_scheduler.py .
COPY run_macro_etf_ingestor.py .
COPY start.sh .
RUN chmod +x start.sh

# start.sh launches both schedulers as background processes so both
# fear_greed/funding (run_scheduler) and macro_releases/etf (run_macro_etf_ingestor)
# are kept fresh without needing a separate container.
CMD ["./start.sh"]
