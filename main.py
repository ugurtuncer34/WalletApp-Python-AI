from fastapi import FastAPI

app = FastAPI(title="FamilyFinance AI & OCR Service")

@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "service": "FamilyFinance AI Core",
        "version": "1.0.0"
    }