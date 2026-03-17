import requests
import os
import glob

# Upload ALL PDFs from docs/ folder
pdf_files = glob.glob(r"C:\Users\rabie\rag-system\docs\*.pdf")

if not pdf_files:
    print("❌ No PDFs found in docs/")
else:
    for pdf_path in pdf_files:
        print(f"Uploading: {pdf_path}")
        with open(pdf_path, "rb") as f:
            response = requests.post(
                "http://localhost:8000/upload",
                files={"file": (os.path.basename(pdf_path), f, "application/pdf")}
            )
        print(response.json())