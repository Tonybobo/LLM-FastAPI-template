from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional
from src.models.llm import create_llm, generate_summary
from src.loaders.web_loader import WebLoader
import asyncio

app = FastAPI(title="Article Summarizer API")

model, tokenizer, device = create_llm()
loader = WebLoader()

class SummaryRequest(BaseModel):
    url: HttpUrl
    custom_prompt: Optional[str] = None

@app.post("/api/summarize")
async def summarize_url(request: SummaryRequest):
    try:
        documents = await loader.load_and_process(str(request.url))
        
        full_text = " ".join([doc.page_content for doc in documents])
        
        summary = generate_summary(model, tokenizer, full_text, device, request.custom_prompt)
        
        return {
            "url": str(request.url),
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}