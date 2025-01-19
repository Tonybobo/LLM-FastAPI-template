from pydantic_settings import BaseSettings , SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    MODEL_ID: str = os.getenv("MODEL_ID" , "mistralai/Mistral-7B-Instruct-v0.3")  # Updated to use the latest Mistral model
    HF_TOKEN:str = os.getenv("HF_TOKEN")
    MAX_LENGTH: int = 4096 
    MIN_LENGTH: int = 50
    TEMPERATURE: float = 0.7
    TOP_P : float = 0.95
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

settings = Settings()