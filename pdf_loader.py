import fitz  # PyMuPDF
import os

def load_pdf_texts(pdf_folder='data'):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            path = os.path.join(pdf_folder, filename)
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append({"filename": filename, "text": text})
    return documents

if __name__ == "__main__":
    docs = load_pdf_texts()
    print(f"Loaded {len(docs)} documents.")
