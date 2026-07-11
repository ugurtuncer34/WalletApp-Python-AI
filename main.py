from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
import os
import json
import io
import pdfplumber
import logging
import time
import asyncio
import contextvars
import uuid
from dotenv import load_dotenv
from openai import AsyncOpenAI

# OPENTELEMETRY IMPORTS
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# ---------------------------------------------------------
# CONFIGURE LOGGING & CORRELATION ID
# ---------------------------------------------------------
correlation_id_var = contextvars.ContextVar("correlation_id", default="unknown")

class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id_var.get()
        return True
    
logger = logging.getLogger("FamilyFinance")
logger.setLevel(logging.INFO)
logger.handlers.clear()

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - [%(correlation_id)s] - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
handler.addFilter(CorrelationIdFilter())
logger.addHandler(handler)

load_dotenv()

# ---------------------------------------------------------
# SECURITY & ENVIRONMENT
# ---------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
NLP_API_SECRET = os.getenv("NLP_API_SECRET", "development_fallback_secret")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != NLP_API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

app = FastAPI(
    title="FamilyFinance AI & NLP Service",
    docs_url=None if ENVIRONMENT == "production" else "/docs",
    redoc_url=None if ENVIRONMENT == "production" else "/redoc",
    openapi_url=None if ENVIRONMENT == "production" else "/openapi.json"
)

# Correlation Middleware
class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        correlation_id_var.set(corr_id)
        
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = corr_id
        return response

app.add_middleware(CorrelationIdMiddleware)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# OPENTELEMETRY TRACING
# ---------------------------------------------------------
otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")

resource = Resource.create({"service.name": "FamilyFinance.NLP"})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Instrument the app AFTER middlewares are added
FastAPIInstrumentor.instrument_app(app)

# ---------------------------------------------------------
# APP LOGIC
# ---------------------------------------------------------
client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    max_retries=1, 
    timeout=300.0   
)

@app.get("/health")
async def health_check():
    return {"status": "active", "service": "FamilyFinance NLP Core"}

async def process_pdf_page_async(page_text: str, page_num: int, system_prompt: str):
    logger.info(f"Page {page_num}: Sending to DeepSeek...")
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-v4-flash",
            response_format={"type": "json_object"}, 
            temperature=0.0, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": page_text}
            ]
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Page {page_num}: Processed in {elapsed:.2f} seconds.")
        
        response_content = response.choices[0].message.content
        ai_result = json.loads(response_content)
        return ai_result.get("transactions", [])
        
    except Exception as e:
        logger.error(f"Page {page_num}: Failed to process. Error: {str(e)}")
        return []

@app.post("/api/nlp/parse-statement")
async def parse_statement(
    file: UploadFile = File(...),
    categories: str = Form(None),
    merchants: str = Form(None),
    api_key: str = Depends(verify_api_key)
):
    try:
        logger.info("==================================================")
        logger.info(f"NEW REQUEST: Parsing statement -> {file.filename}")
        
        parsed_categories = []
        parsed_merchants = []
        
        if categories:
            try:
                parsed_categories = json.loads(categories)
            except json.JSONDecodeError:
                parsed_categories = [c.strip() for c in categories.split(",") if c.strip()]
                logger.info("Categories: Fell back to comma-separated parsing.")
            logger.info(f"Loaded {len(parsed_categories)} categories.")

        if merchants:
            try:
                parsed_merchants = json.loads(merchants)
            except json.JSONDecodeError:
                logger.warning("Merchants: Failed to parse JSON. Complex fallback not possible, ignoring.")
            logger.info(f"Loaded {len(parsed_merchants)} merchants.")

        logger.info("STEP 1: Extracting text using pdfplumber...")
        pages_text = []
        
        pdf_bytes = await file.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted and len(extracted) > 50:
                    pages_text.append(extracted)

        logger.info(f"STEP 1 COMPLETE: Extracted {len(pages_text)} valid pages.")

        if not pages_text:
            return {"success": False, "message": "Could not extract text. PDF might be empty."}

        system_prompt = f"""
        You are a highly precise financial data extraction API. 
        You will receive the raw text of ONE PAGE from a Turkish credit card statement.
        Extract all individual expenditures/purchases found on this page.

        KNOWN DATA FROM DATABASE:
        Subcategories: {json.dumps(parsed_categories, ensure_ascii=False)}
        Merchants: {json.dumps(parsed_merchants, ensure_ascii=False)}

        CRITICAL "FUZZY MATCHING" RULES FOR MERCHANTS:
        Credit card statements often contain POS terminal codes, store locations, or prefixes (e.g., "OPET SS CORLU", "CACEL FASHION", "GOOGLE *YOUTUBE", "FİLE MARKET MAĞAZACILIK").
        You must perform SMART PARTIAL MATCHING between the statement's raw description and the 'name' field in the Known Merchants list.

        1. IF IT MATCHES A KNOWN MERCHANT (even partially):
           - "merchant": You MUST use the exact, clean 'name' from the Known Merchants list (e.g., output "OPET", NOT "OPET SS CORLU"). Never use the raw text if a match is found.
           - "category": Use the exact 'defaultCategoryName' of that matched merchant.
           - "isMerchantMatched": true

        2. IF IT DOES NOT MATCH ANY KNOWN MERCHANT:
           - "merchant": The raw description exactly as written on the statement.
           - "category": Guess the most appropriate category ONLY from the Known Subcategories list.
           - "isMerchantMatched": false

        3. Ignore payments, limits, summaries, and mil/points information.

        You MUST respond ONLY with a JSON object containing a "transactions" array. 
        Structure:
        {{
            "transactions": [
                {{
                    "date": "YYYY-MM-DD",
                    "merchant": "string",
                    "amount": float,
                    "category": "string",
                    "isMerchantMatched": boolean
                }}
            ]
        }}
        """
        
        logger.info(f"STEP 2: Dispatching {len(pages_text)} concurrent API requests...")
        api_start_time = time.time()
        
        tasks = [
            process_pdf_page_async(text, i + 1, system_prompt) 
            for i, text in enumerate(pages_text)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"STEP 2 COMPLETE: Parallel processing finished in {time.time() - api_start_time:.2f} seconds.")

        logger.info("STEP 3: Aggregating chunked JSON arrays...")
        all_transactions = []
        
        for res in results:
            if isinstance(res, list):
                all_transactions.extend(res)
        
        all_transactions.sort(key=lambda x: x.get("date", ""))
        
        logger.info(f"SUCCESS: Aggregated a total of {len(all_transactions)} transactions!")
        logger.info("==================================================")

        return {
            "success": True,
            "filename": file.filename,
            "total_transactions": len(all_transactions),
            "data": all_transactions,
        }

    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Statement Parsing Error: {str(e)}"
        }