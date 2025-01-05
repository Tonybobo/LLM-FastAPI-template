from pydantic_settings import BaseSettings , SettingsConfigDict
import os

class Settings(BaseSettings):
    MODEL_ID: str = os.getenv("MODEL_ID" , "facebook/bart-large-cnn")  # Updated to use the latest Mistral model
    HF_TOKEN:str = os.getenv("HUGGING_FACE_TOKEN")
    MAX_LENGTH: int = 2048
    TEMPERATURE: float = 0.7
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

settings = Settings()