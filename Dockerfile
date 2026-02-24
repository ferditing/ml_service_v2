FROM python:3.11-slim

WORKDIR /app

# Install build deps only if needed by some wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
