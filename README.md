# Multilingual-RAG
Simple Multilingual RAG System

# Multilingual RAG System for Bangla & English Queries

## Project Overview

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline capable of understanding and responding to user queries in both Bangla and English. The system fetches relevant information from a PDF document corpus (Bangla Book - *HSC26 Bangla 1st paper*) and generates meaningful answers grounded in the retrieved content using a Bangla T5 model.

---

## Setup Guide

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Multilingual-RAG

    Create and activate Python virtual environment:

python3 -m venv env
source env/bin/activate

Install required packages:

    pip install -r requirements.txt

    Run the scripts:

        Extract & chunk PDF text: python scripts/chunk_texts.py

        Generate embeddings: python scripts/embed_chunks.py

        Search chunks: python scripts/search_chunks.py

        Interactive QA inference: python scripts/qa_infer.py

Tools, Libraries & Packages Used

    PyMuPDF (fitz): For PDF text extraction, chosen for robust handling of Bangla text and layout.

    Transformers (HuggingFace): For loading and using the pre-trained Bangla T5 model (csebuetnlp/banglat5).

    PyTorch: Backend for the transformer model.

    SentenceTransformers or other embedding tools (custom or HuggingFace models): For generating semantic vector embeddings.

    tqdm: Progress bar visualization.

    re (regex): For cleaning and filtering Bangla text.

    FAISS or simple in-memory cosine similarity (planned/implied).

Sample Queries and Outputs
Bangla Query

Question: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
Retrieved Context Snippet: (Chunk containing related Bangla text from the PDF)
Generated Answer: (Model-generated Bangla response based on retrieved context)
English Query

Question: Who was the first president of Bangladesh?
Retrieved Context Snippet: (Relevant Bangla text chunk about Bangladesh history)
Generated Answer: (Model-generated answer)
API Documentation

Currently, the system is a CLI-based interactive script without a deployed API.
Future plans include wrapping the QA inference in a REST API (e.g., FastAPI) for external integration.
Evaluation Matrix

Evaluation not yet formally implemented.
Plans include human evaluation of answer relevance and coherence, and automated metrics such as BLEU, ROUGE for generated answers compared to reference answers.
FAQ / Key Questions
What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

    Used PyMuPDF (fitz) to extract raw text from PDF because it preserves layout and works well with Unicode Bangla text.

    Faced challenges with mixed content: Bangla text, tables, and some images cause noise and formatting issues in raw extraction, requiring careful cleaning and chunking.

What chunking strategy did you choose? Why do you think it works well for semantic retrieval?

    Initially chunked the document into large single chunks based on extracted paragraphs.

    Current plan is to refine to smaller, roughly 300-token chunks for better semantic granularity and retrieval precision.

    Chunking balances capturing enough context and enabling effective similarity comparison.

What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

    Used HuggingFace transformer-based sentence embeddings compatible with Bangla text.

    The model produces dense vector representations encoding semantic meaning beyond simple lexical matching.

    Embeddings enable semantic similarity search, improving retrieval even if exact wording differs.

How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

    Used cosine similarity between query embedding and chunk embeddings to rank relevance.

    Stored embeddings in memory or simple files for prototype stage; scalable vector databases like FAISS planned for future.

    Cosine similarity is a common, efficient metric for semantic similarity in vector space.

How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

    Cleaning and preprocessing both queries and chunks help reduce noise.

    Using semantic embeddings allows the system to match based on meaning rather than exact words.

    Vague queries may retrieve less relevant chunks, reducing answer quality; improving chunk granularity and context length helps mitigate this.

Do the results seem relevant? If not, what might improve them?

    Current results are promising but limited by chunk size and model tuning.

    Improvements can come from:

        Better chunking strategy (smaller, semantically coherent chunks)

        Larger or fine-tuned embedding models for Bangla

        Incorporation of recent chat context (short-term memory)

        Prompt engineering and answer post-processing

        Expanding the document corpus for richer knowledge

Summary

This project demonstrates a foundational RAG pipeline for Bangla and English queries with:

    PDF text extraction and cleaning

    Document chunking and embedding

    Similarity search over chunk embeddings

    Answer generation using a BanglaT5 model

It sets the stage for further enhancements in multi-turn dialogue, improved chunking, and deployment.