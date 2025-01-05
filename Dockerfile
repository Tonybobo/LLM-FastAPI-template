FROM nvcr.io/nvidia/pytorch:24.02-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (excluding torch as it's already in base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit config
COPY .streamlit /app/.streamlit

# Copy source code and ensure proper structure
COPY src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app

# Create cache directory for Hugging Face
RUN mkdir -p /root/.cache/huggingface

# Expose both Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000

# Copy and set permissions for run script
COPY run.py .
RUN chmod +x run.py

# Run both services
CMD ["python", "run.py"]