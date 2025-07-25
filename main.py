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

generator = Generator()

# Accept user query
while True:
    query = input("Enter your question (English or Bengali), or 'exit' to quit: ")
    if query.lower() in ["exit", "quit"]:
        break
    query_embedding = embedder.embed([query]).astype("float32")
    top_docs = retriever.search(query_embedding, top_k=2)
    combined_context = " ".join([doc["text"].strip().replace("\n", " ")[:1000] for doc in top_docs])
    print("----Context being passed to model----")
    print(combined_context[:500])  # just show first 500 characters
    answer = generator.generate(query, combined_context)
    print("\nAnswer:")
    print(answer)
    print("-" * 30)


#Sample Test Case:
#User Question: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
#Expected Answer: শুম্ভুনাথ
#User Question: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
#Expected Answer: মামাকে
#User Question: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
#Expected Answer: ১৫ বছর
