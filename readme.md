# Article Summarizer

A web application that summarizes articles using state-of-the-art NLP models. Built with Streamlit for the frontend and FastAPI for the backend API, this application provides both a user interface and API endpoints for article summarization.

## Features

- Web interface for easy article summarization
- RESTful API endpoints for integration
- GPU acceleration support for faster processing
- Custom prompt support for targeted summarization
- Real-time processing with progress indicators
- Supports both UI and API access

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **ML Model**: BART-large-CNN
- **Container**: Docker with NVIDIA GPU support
- **Python**: 3.9
- **Additional Libraries**: 
 - Transformers
 - PyTorch
 - Pandas
 - Beautiful Soup 4

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA 12.4+ support
- NVIDIA Container Toolkit installed

### Installing NVIDIA Container Toolkit

```bash
# Add NVIDIA's official repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


project_root/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py      # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm.py          # ML model initialization
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── web_loader.py   # Article loading utilities
│   ├── streamlit_app.py    # Streamlit frontend
├── .streamlit/             # Streamlit configuration
├── run.py                  # Application runner
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
└── docker-compose.yml     # Container orchestration

```



# Build the Docker container:

```

docker-compose build --no-cache
docker-compose up

```
The application will be available at:

- Web UI: http://localhost:8501
- API: http://localhost:8000

# API Documentation: http://localhost:8000/docs

## API Usage
- Summarize Article
```
POST /api/summarize
{
    "url": "https://example.com/article",
    "custom_prompt": "Optional custom summarization prompt"
}
```

- Health Check

```
GET /api/health
```
