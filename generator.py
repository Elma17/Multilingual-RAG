from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name="google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, query, context):
        input_text = f"question: {query} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
