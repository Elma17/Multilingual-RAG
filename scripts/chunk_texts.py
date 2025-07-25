import fitz  # PyMuPDF
import re
import json

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    return full_text

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    return text.strip()

def chunk_text_by_tokens(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def save_chunks_as_json(chunks, file_path="chunks.json"):
    chunk_dicts = [{"text": chunk} for chunk in chunks]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(raw_text)

    # Use this line instead of paragraph chunking:
    chunks = chunk_text_by_tokens(clean, max_tokens=30)

    print(f"Extracted {len(chunks)} chunks")
    print("Sample:", chunks[0][:300])
    
    save_chunks_as_json(chunks)

