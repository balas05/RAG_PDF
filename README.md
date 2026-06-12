# Terminal RAG Assistant 🤖

A locally-running Retrieval-Augmented Generation (RAG) system that reads PDF documents, stores them in a local vector database, and allows you to chat with them directly from your terminal using Google Gemini.

## Features
* **Local Document Ingestion**: Scans the `data/pdfs` folder, reads your PDFs, and splits them into searchable chunks.
* **Local Vector Database**: Uses `ChromaDB` and open-source `SentenceTransformers` (`all-MiniLM-L6-v2`) to mathematically map your documents entirely offline.
* **AI Chat**: Uses Google's `gemini-2.5-flash` via the new GenAI SDK to synthesize answers based **only** on your documents (no hallucinations!).

## Setup Instructions

**1. Install Dependencies**
```bash
pip install -r requirements.txt
pip install langchain-google-genai google-generativeai python-dotenv
```

**2. Add your API Key**
Create a `.env` file in the root folder and add your Google Gemini key:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```

**3. Ingest Your Documents**
Place any `.pdf` files you want to chat with inside the `data/pdfs/` directory. Then, run the database builder:
```bash
python build_vector_db.py
```

**4. Start Chatting!**
Once the database is built, launch the terminal chatbot:
```bash
python query_rag.py
```
Type your question, and the AI will search your PDFs for the answer!
