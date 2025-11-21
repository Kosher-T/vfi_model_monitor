# Use a lightweight Python version
FROM python:3.9-slim

# 1. Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the working directory inside the container
WORKDIR /app

# 3. Install Python libraries (The SLOW step - CACHED)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your scripts and baseline memory
COPY drift_detector.py .
COPY drift_analyzer.py .
COPY monitoring_service.py .
COPY embeddings/baseline_embeddings.npy . 

# 5. Create directories for data and output
RUN mkdir /app/incoming_data
RUN mkdir /app/status_output

# 6. The command to run when the container starts
CMD ["python", "monitoring_service.py"]