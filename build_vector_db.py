import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF to read PDF files

def main():
    
    import os
    print("=== Step 1 & 2: Loading and Chunking Data ===")
    pdf_dir = "data/pdfs"
    
    # We split the long text into larger, overlapping chunks to keep tables intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200 
    )
    
    all_chunks = []
    all_metadatas = []
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Reading PDF: {file_path}")
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                text += doc[page_num].get_text() + "\n"
                
            # Chunk this specific document
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            
            # Save the filename as metadata so we know where this chunk came from
            all_metadatas.extend([{"source": filename}] * len(chunks))
        
    print(f"\nSuccessfully extracted and split documents into {len(all_chunks)} total chunks.")
    for i, chunk in enumerate(all_chunks[:2]):
        print(f"  - Chunk {i+1}: {chunk[:50]}...")

    print("\n=== Step 3: Embeddings & Vector DB ===")
    print("Downloading/Loading high-quality Embedding Model (BAAI/bge-base-en-v1.5)...")
    # This model is much better for large PDFs
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    # Initialize ChromaDB in our current folder
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        chroma_client.delete_collection("my_documents")
        print("Deleted old database structure to make way for new dimensions...")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="my_documents")
    
    print(f"Converting {len(all_chunks)} chunks to math and saving to database...")
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    
    # Calculate embeddings
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True).tolist()
    
    # Upsert in batches to avoid Chroma's max batch size limit
    batch_size = 5000
    print("Upserting to database in batches...")
    for i in range(0, len(all_chunks), batch_size):
        end_idx = i + batch_size
        print(f"  Upserting batch {i} to {min(end_idx, len(all_chunks))}...")
        collection.upsert(
            documents=all_chunks[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=all_metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
    print("Success! Data is now in the Vector DB.")

if __name__ == "__main__":
    main()
