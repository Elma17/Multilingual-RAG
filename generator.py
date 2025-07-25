from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name="google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, query, context):
        input_text = f"Answer the question based on the context below:\nContext: {context}\nQuestion: {query}\nAnswer:"
        print("DEBUG: Input to model:")
        print(input_text)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

