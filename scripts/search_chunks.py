import numpy as np
from sentence_transformers import SentenceTransformer
import json

def load_embeddings(file_path="chunk_embeddings.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    embeddings = np.array([item["embedding"] for item in data])
    return texts, embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_top_k(query, texts, embeddings, model, k=3):
    query_emb = model.encode([query])[0]
    scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [(texts[i], scores[i]) for i in top_k_indices]

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts, embeddings = load_embeddings()

    query = input("Enter query: ")
    results = retrieve_top_k(query, texts, embeddings, model, k=3)
    print(f"Top {len(results)} results:")
    for i, (text, score) in enumerate(results):
        print(f"\nRank {i+1}, Score: {score:.4f}")
        print(text[:500])
