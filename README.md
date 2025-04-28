# LEAGEN
Built a secure AI system using RAG (LangChain + FAISS) to automate legal document summarization and multilingual Q&amp;A via a Gradio interface, reducing manual legal research time by over 60%.


# LEGALGEN: A Multimodal AI System for Legal Summarization Using Generative AI

# Project Overview
LEGALGEN is a secure, AI-powered system designed to automate legal document summarization and legal question answering (Q&A) using Retrieval-Augmented Generation (RAG) and Generative AI techniques. It provides multilingual support for over 11 Indian languages through a simple Gradio interface, significantly reducing manual legal research time while preserving data privacy.


# Features
Retrieval-Augmented Generation (RAG) pipeline for intelligent legal document retrieval and summarization.

Support for multilingual text and voice queries via Gemini API (e.g., Tamil, Hindi, English).

Secure, locally processed PDF document handling with PyMuPDF.

Semantic search using HuggingFace Embeddings (all-MiniLM-L6-v2) and FAISS vector store.

Auto-generated summaries and audio files (text-to-speech) for accessibility.

Scalable modular design for future integration with dynamic legal databases.

# Tools and Technologies Used

LangChain, FAISS, HuggingFace Embeddings, Gemini API, Gradio, PyMuPDF, RecursiveCharacterTextSplitter, gTTS, Python, dotenv.

# Project Structure
├── legal_doc_summ.py         # Main backend logic for summarization, RAG setup, translation, audio generation
├── LEGALGEN.pdf              # Project documentation (report)
├── output_files/             # Generated PDFs and audio files
├── faiss_index/              # Local vector database (semantic index)
├── .env                      # Environment variables (API keys)
├── README.md                 # Project overview

