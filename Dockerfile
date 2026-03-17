# Use official Python slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    espeak \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install video + TTS packages
RUN pip install --no-cache-dir "moviepy==1.0.3" Pillow edge-tts asyncio

# Copy application code
COPY api.py .
COPY rag.py .
COPY video.py .

# Create folders
RUN mkdir -p docs chroma_db videos

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]