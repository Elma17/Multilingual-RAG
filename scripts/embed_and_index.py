from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def embed_and_index_chunks(chunks):
    # Load embedding model (multilingual)
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # Embed chunks (list of strings)
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Convert embeddings to float32 numpy array (FAISS requires this)
    embeddings = np.array(embeddings).astype('float32')
    
    # Build FAISS index (use inner product or L2)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance
    
    # Add vectors to index
    index.add(embeddings)
    
    print(f"Indexed {index.ntotal} chunks.")
    
    return index, model, chunks

# Usage example:
if __name__ == "__main__":
    # Suppose you already have chunks extracted from your PDF
    chunks = [
        "বাংলা সাহিত্যের পরিচয়",
        "অনুপমের ভাষায় সুপুরুষ ...",
        "কল্যাণীর প্রকৃত বয়স ছিল ...",
        # more chunks ...
    ]
    
    index, model, chunks = embed_and_index_chunks(chunks)
