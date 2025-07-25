from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.model_name = "csebuetnlp/banglat5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)


    def generate(self, question, context):
        prompt = f"প্রসঙ্গ: {context}\nপ্রশ্ন: {question}\nউত্তর:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(**inputs, max_length=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

