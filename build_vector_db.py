import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF to read PDF files

def main():
    import os
    print("=== Step 1: Loading Data ===")
    pdf_dir = "data/pdfs"
    text = ""
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Reading PDF: {file_path}")
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                text += doc[page_num].get_text() + "\n"
        
    print(f"Successfully extracted text from your PDF(s).")

    print("\n=== Step 2: Chunking ===")
    # We split the long text into smaller, overlapping chunks
    # This ensures we don't feed too much text to the LLM at once later
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Increased for larger PDF documents
        chunk_overlap=100 
    )
    chunks = text_splitter.split_text(text)
    print(f"Split document into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:2]):
        print(f"  - Chunk {i+1}: {chunk[:50]}...")

    print("\n=== Step 3: Embeddings & Vector DB ===")
    print("Downloading/Loading local Embedding Model (this runs locally!)...")
    # This free model converts text into an array of 384 numbers (the vector)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB in our current folder
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="my_documents")
    
    print("Converting chunks to math and saving to database...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_path} for _ in range(len(chunks))]
    
    # Calculate embeddings and save to database
    embeddings = embedding_model.encode(chunks).tolist()
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("Success! Data is now in the Vector DB.")

if __name__ == "__main__":
    main()
