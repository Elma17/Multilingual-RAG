import faiss
import numpy as np

class Retriever:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.docs = []

    def add(self, embeddings, documents):
        self.index.add(embeddings)
        self.docs = documents

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.docs[i] for i in indices[0]]
