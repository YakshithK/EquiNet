# pdf_extractor.py

import pdfplumber
import json
import os
from langdetect import detect

PDF_FOLDER = "./pdf_reports/"
OUTPUT_FILE = "pdf_dataset.json"

def extract_pdf_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        return text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract {pdf_path}: {e}")
        return None

def process_pdfs():
    data = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            text = extract_pdf_text(path)
            if text:
                entry = {
                    "source": "PDF Report",
                    "file_name": filename,
                    "text": text,
                    "language": detect(text),
                    "date": None  # optional: add if known
                }
                data.append(entry)
    return data

if __name__ == "__main__":
    extracted_data = process_pdfs()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Extracted {len(extracted_data)} PDF reports into {OUTPUT_FILE}")
