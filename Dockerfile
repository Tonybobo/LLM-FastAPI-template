FROM nvcr.io/nvidia/pytorch:23.12-py3

WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Remove torch from requirements as it's already installed
RUN sed -i '/torch==/d' requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for model cache
RUN mkdir -p /app/src/models/bart

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_LOCAL_DIR=/app/local_models/distilbart

# Expose the port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]