from pydantic_settings import BaseSettings , SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    MODEL_ID: str = os.getenv("MODEL_ID", "sshleifer/distilbart-cnn-12-6")
    MODEL_LOCAL_DIR: str = os.getenv("MODEL_LOCAL_DIR", "src/models/bart")
    MAX_LENGTH: int = 1024
    MIN_LENGTH: int = 100
    MAX_SUMMARY_LENGTH: int = 512
    MIN_SUMMARY_LENGTH: int = 100
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    NUM_BEAMS: int = 4  
    LENGTH_PENALTY: float = 2.0
    S3_BUCKET:str = os.getenv("S3_BUCKET" , "")
    AWS_ACCESS_KEY_ID:str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY:str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION" , "ap-southeast-1")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

settings = Settings()