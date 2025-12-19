# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Set environment variables
# PYTHONPATH=/app ensures Python can find 'src' and 'app' modules
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 4. Copy dependency file and install
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy the source code folders
# We mirror your local structure: /app/src, /app/app, /app/artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY model_build/ /app/model/

RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# 8. Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
