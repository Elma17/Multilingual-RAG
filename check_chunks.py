import json

def load_chunks(path="chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

chunks = load_chunks()

print(f"Total chunks: {len(chunks)}")
print("Sample chunk text (first 500 chars):")
print(chunks[0]["text"][:500])
