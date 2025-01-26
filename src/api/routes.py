from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional
from src.loaders.web_loader import WebLoader
from src.utils.logger import get_logger
from src.models.model_manager import create_llm 

app = FastAPI(title="Article Summarizer API")

bart_manager = create_llm()
loader = WebLoader()
logger = get_logger()

class SummaryRequest(BaseModel):
    url: HttpUrl
    custom_prompt: Optional[str] = None

@app.post("/api/summarize")
async def summarize_url(request: SummaryRequest):
    try:
        logger.info(f"Request ====> url: {request.url} , prompt: {request.custom_prompt} ")

        documents = await loader.load_and_process(str(request.url))
        
        full_text = " ".join([doc.page_content for doc in documents])
        
        summary = bart_manager.generate_summary(full_text , request.custom_prompt) 

        logger.info(f" response ====> url: {request.url} , summary : {summary}")
        
        return {
            "url": str(request.url),
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"summarize_url function err : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}