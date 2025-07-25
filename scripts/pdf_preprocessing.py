import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text += text + "\n"
    return full_text

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = text.strip()
    return text

def chunk_text_by_paragraph(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def chunk_text_by_tokens(text, max_tokens=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

def save_chunks(chunks, filename="chunks.json"):
    chunk_dicts = [{"text": chunk} for chunk in chunks]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {filename}")

if __name__ == "__main__":
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(raw_text)

    # Choose one chunking strategy
    chunks = chunk_text_by_paragraph(clean)
    # or chunks = chunk_text_by_tokens(clean, max_tokens=300)

    print(f"Extracted {len(chunks)} chunks")
    print(chunks[0])

    # Save chunks for later use
    save_chunks(chunks)

