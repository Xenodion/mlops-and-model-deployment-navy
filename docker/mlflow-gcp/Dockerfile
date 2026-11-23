# Shared MLflow tracking server for Cloud Run (private service, one instance, scale to zero).
# Same MLflow version as the local dev stack; GCS driver for artifacts. Python 3.12: wheels for every dependency.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.17.2 google-cloud-storage==3.4.0

COPY serve.py /app/serve.py
ENV PORT=8080
EXPOSE 8080
CMD ["python", "/app/serve.py"]
