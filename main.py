from pdf_loader import load_pdf_texts
from embedder import Embedder
from retrieval import Retriever
from generator import Generator
import numpy as np

# Load and embed documents
documents = load_pdf_texts()
texts = [doc["text"] for doc in documents]

embedder = Embedder()
embeddings = embedder.embed(texts)
embeddings = np.array(embeddings).astype("float32")

# Build index
retriever = Retriever(embedding_dim=embeddings.shape[1])
retriever.add(embeddings, documents)

# Accept user query
query = input("Enter your question (English or Bengali): ")
query_embedding = embedder.embed([query]).astype("float32")

# Retrieve top docs
top_docs = retriever.search(query_embedding, top_k=2)
combined_context = " ".join([doc["text"] for doc in top_docs])

# Generate answer
generator = Generator()
answer = generator.generate(query, combined_context)

print("\nAnswer:")
print(answer)
