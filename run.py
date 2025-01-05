import subprocess
import sys
from multiprocessing import Process

def check_cuda():
    import torch
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

def run_streamlit():
    subprocess.run([
        "streamlit", "run", "src/streamlit_app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501"
    ])

def run_fastapi():
    subprocess.run([
        "uvicorn", "src.api.routes:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

if __name__ == "__main__":

    check_cuda()

    # Start FastAPI process
    api_process = Process(target=run_fastapi)
    api_process.start()
    
    # Start Streamlit process
    streamlit_process = Process(target=run_streamlit)
    streamlit_process.start()
    
    # Wait for both processes
    streamlit_process.join()
    api_process.join()