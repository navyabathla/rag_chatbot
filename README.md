## Pentair Product Chatbot (RAG-based)
---
### Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot designed for Pentair products. The chatbot can answer product-related queries using context retrieved from product manuals and documents stored in a ChromaDB vector database.
Additionally, it logs user information as leads (including name, email, location, and product type) in Firebase Firestore.

The backend is built using Flask, LangChain, OpenRouter (Mixtral or LLaMA3 model), and SentenceTransformerEmbeddings for semantic search.
---
## Features
RAG-based response generation using product PDFs.

Session management: Stores chat history per user in Firebase.

Lead collection: Detects product type, name, and user location automatically.

Firebase Firestore integration for user sessions and lead storage.

Web-based UI using Flask templates (HTML, CSS, JavaScript).

Vector database (ChromaDB) for document embeddings.
---

## Tech Stack
Backend: Python, Flask, LangChain, OpenRouter API

Embeddings: SentenceTransformer

Database: ChromaDB (local vector DB)

Storage: Firebase Firestore

Frontend: HTML, CSS, JavaScript (Flask static/ and templates/)
---
