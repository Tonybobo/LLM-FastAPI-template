from pydantic_settings import BaseSettings , SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    MODEL_ID: str = os.getenv("MODEL_ID" , "google/pegasus-cnn_dailymail") 
    HF_TOKEN:str = os.getenv("HF_TOKEN")
    MAX_LENGTH: int = 1024 
    MIN_LENGTH: int = 30
    TEMPERATURE: float = 0.7
    TOP_P : float = 0.95
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

settings = Settings()