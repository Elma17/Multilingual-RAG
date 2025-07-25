import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def clean_bangla_text(text):
    # Keep Bangla Unicode range chars, spaces, punctuation
    allowed_chars = re.findall(r'[\u0980-\u09FF\s.,?!“”"\':;()\-]+', text)
    return ''.join(allowed_chars).strip()

class Generator:
    def __init__(self):
        self.model_name = "csebuetnlp/banglat5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def generate(self, question, context):
        question_clean = clean_bangla_text(question)
        context_clean = clean_bangla_text(context)

        prompt = f"প্রশ্ন: {question_clean}\nপ্রাসঙ্গিক তথ্য: {context_clean}\nউত্তর:"

        print("---- Prompt to model ----")
        print(prompt)
        print("------------------------")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        output_ids = self.model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_p=0.9,
        )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = answer.replace(prompt, "").strip()
        return answer


if __name__ == "__main__":
    generator = Generator()

    while True:
        question = input("Enter your question (Bangla or English), or 'exit' to quit: ").strip()
        if question.lower() == "exit":
            break

        # For now, you can pass full document text as context or chunks from retrieval
        context = "অনলাইন ব্যাচ সম্পর্কিত যেককাকনা জিজ্ঞাাসা ... (your cleaned text chunks here)"

        answer = generator.generate(question, context)
        print("\nAnswer:")
        print(answer)
        print("-" * 30)
