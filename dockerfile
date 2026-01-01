# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
COPY templates ./templates

COPY model /app/model

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
