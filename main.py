from fastapi import FastAPI, UploadFile, File
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables securely from .env
load_dotenv()

app = FastAPI(title="FamilyFinance AI & NLP Service")

# Initialize the DeepSeek API client asynchronously
client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

@app.get("/health")
async def health_check():
    return {"status": "active", "service": "FamilyFinance NLP Core"}

# ---------------------------------------------------------
# NEXT PHASE: Bank Statement (PDF/CSV) Parsing Engine
# ---------------------------------------------------------
@app.post("/api/nlp/parse-statement")
async def parse_statement(file: UploadFile = File(...)):
    # The uploaded file (e.g., Garanti Bank statement) will be processed here
    # using Pandas or PDF tools, and then interpreted by the DeepSeek NLP engine.
    
    return {
        "success": True,
        "message": "NLP engine is ready for bank statement processing.",
        "filename": file.filename
    }