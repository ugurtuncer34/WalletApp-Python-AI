from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import pytesseract

app = FastAPI(title="FamilyFinance AI & OCR Service")

@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "service": "FamilyFinance AI Core",
        "version": "1.0.0"
    }

@app.post("/api/ocr/receipt")
async def read_receipt(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    new_height = 1500
    ratio = new_height / height
    new_width = int(width * ratio)
    resized_img = cv2.resize(img, (new_width, new_height))

    # Convert to Grayscale (Tesseract LSTM engine prefers grayscale over hard binary)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Increase Contrast and Brightness slightly instead of harsh thresholding
    # alpha = 1.2 (contrast control), beta = 0 (brightness control)
    enhanced_img = cv2.convertScaleAbs(gray_img, alpha=1.2, beta=0)

    # Run Tesseract with custom config
    # --oem 3 : Default LSTM engine
    # --psm 4 : Assume a single column of text of variable sizes (Perfect for receipts)
    custom_config = r'--oem 3 --psm 4'
    extracted_text = pytesseract.image_to_string(enhanced_img, lang='tur', config=custom_config)

    return {
        "filename": file.filename,
        "raw_text": extracted_text.strip()
    }