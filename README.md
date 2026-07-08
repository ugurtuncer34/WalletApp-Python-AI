# FamilyFinance AI & NLP Service

A high-performance, asynchronous Natural Language Processing (NLP) microservice designed to parse complex Turkish credit card statements (PDFs) and extract structured transaction data using Large Language Models (LLMs).

## 🚀 Architecture & Key Features

- **Framework:** FastAPI (Python)
- **AI Engine:** DeepSeek API (`deepseek-v4-flash`) via `AsyncOpenAI` client.
- **PDF Extraction:** `pdfplumber` for precise text-layer scraping.
- **Asynchronous Chunking:** Processes multiple PDF pages simultaneously using `asyncio.gather`. This bypasses strict LLM `max_tokens` limits (8192) and prevents HTTP timeouts on massive documents.
- **Fuzzy Entity Resolution:** Implements an advanced System Prompt that takes a JSON array of known merchants and categories from the frontend. It performs smart partial matching to clean raw POS terminal descriptions (e.g., matching "STARBUCKS SS YORK" strictly to "STARBUCKS") and sets an `isMerchantMatched` boolean flag for the client UI.

## 🛠️ Setup & Installation

1. Clone the repository and create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install fastapi uvicorn pdfplumber openai python-dotenv python-multipart
   ```

3. Create a `.env` file in the root directory and add your API key:
   ```env
   DEEPSEEK_API_KEY=your_secure_api_key_here
   ```

4. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

## 📡 API Endpoints

### `POST /api/nlp/parse-statement`

Accepts a PDF file and optional JSON strings for mapping. 

**Form Data Parameters:**
- `file`: The uploaded PDF statement.
- `categories`: (Optional) JSON string array of known categories. Example: `["HEALTH", "FOOD", "BILLS"]`
- `merchants`: (Optional) JSON string array of known merchants. Example: `[{"name": "STARBUCKS", "defaultCategoryName": "COFFEE"}]`

**Response Example:**
```json
{
  "success": true,
  "filename": "statement.pdf",
  "total_transactions": 140,
  "data": [
    {
      "date": "2026-06-16",
      "merchant": "STARBUCKS",
      "amount": 20,
      "category": "COFFEE",
      "isMerchantMatched": true
    }
  ]
}
```