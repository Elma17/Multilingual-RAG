import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import json

def clean_bangla_text(text):
    allowed_chars = re.findall(r'[\u0980-\u09FF\s.,?!“”"\':;()\-]+', text)
    return ''.join(allowed_chars).strip()

def load_embeddings(file_path="chunk_embeddings.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    embeddings = np.array([item["embedding"] for item in data])
    return texts, embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_top_k(query, texts, embeddings, embed_model, k=3):
    query_emb = embed_model.encode([query])[0]
    scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [texts[i] for i in top_k_indices]

class Generator:
    def __init__(self):
        self.model_name = "csebuetnlp/banglat5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def generate(self, question, context):
        question_clean = clean_bangla_text(question)
        context_clean = clean_bangla_text(context)
        prompt = f"প্রশ্ন: {question_clean}\nপ্রাসঙ্গিক তথ্য: {context_clean}\nউত্তর:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        output_ids = self.model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()

if __name__ == "__main__":
    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts, embeddings = load_embeddings()

    generator = Generator()

    while True:
        question = input("Enter your question (Bangla or English), or 'exit' to quit: ").strip()
        if question.lower() == "exit":
            break

        relevant_chunks = retrieve_top_k(question, texts, embeddings, embed_model, k=3)
        context = "\n\n".join(relevant_chunks)

        answer = generator.generate(question, context)
        print("\nAnswer:")
        print(answer)
        print("-" * 40)
