FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application and train the synthetic model bundle at build time.
# The service never trains; it loads models/synthetic at startup.
COPY app/ app/
RUN python -m app.train --out models/synthetic
ENV MODEL_BUNDLE_DIR=/app/models/synthetic

# FastAPI port
EXPOSE 8000

# One worker: a single simulated engine shared by all clients (see app/simulation_service.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
