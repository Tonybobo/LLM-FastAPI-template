import subprocess
import sys
import os;
from multiprocessing import Process
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_cuda():
    import torch
    logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")

def run_streamlit():
    logger.info(f"Starting Streamlit server...")
    try:
      env = os.environ.copy()
      env["PYTHONPATH"] = project_root
      result = subprocess.run([
        "streamlit", "run", "src/streamlit_app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501"
     ], check= True, capture_output=True , text= True , env= env)
      logger.info(f"Streamlit output :{result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit failed to start: {e}")
        logger.error(f"Stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"Error starting Streamlit: {e}")


def run_fastapi():
    logger.info(f"Starting FastApi server...")
    try:
        subprocess.run([
            "uvicorn", "src.api.routes:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])

    except subprocess.CalledProcessError as e:
        logger.error(f"FastApi failed to start: {e}")
        logger.error(f"Stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"Error starting FastApi: {e}")

if __name__ == "__main__":

    check_cuda()

    api_process = Process(target=run_fastapi)
    api_process.start()
    logger.info("FastAPI process started")
    
    streamlit_process = Process(target=run_streamlit)
    streamlit_process.start()
    logger.info("Streamlit Process Started")
    
    try:
        streamlit_process.join()
        api_process.join()
    except KeyboardInterrupt:
        logger.info("Shutting Down")
        streamlit_process.terminate()
        api_process.terminate()
        streamlit_process.join()
        api_process.join()