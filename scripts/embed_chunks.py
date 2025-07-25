import json
from sentence_transformers import SentenceTransformer

def load_chunks(file_path="chunks.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

def save_embeddings(chunks, embeddings, out_file="chunk_embeddings.json"):
    out_data = []
    for chunk, emb in zip(chunks, embeddings):
        out_data.append({
            "text": chunk["text"],
            "embedding": emb.tolist()
        })
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

def main():
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    chunks = load_chunks()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    save_embeddings(chunks, embeddings)
    print(f"Saved embeddings for {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
