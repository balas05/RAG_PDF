import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

def main():
    # 1. Initialize the embedding model to embed the user's question
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Connect to our Vector DB
    print("Connecting to Vector DB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="my_documents")
    
    # 3. Initialize the LLM (Gemini in this case)
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is missing! Please add it to a .env file.")
        return
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    print("\n" + "="*50)
    print("🤖 RAG System Ready! Type 'exit' to quit.")
    print("="*50 + "\n")
    
    while True:
        question = input("You: ")
        if question.lower() in ['exit', 'quit']:
            break
            
        print("\n[System] Searching database for relevant info...")
        # A. Convert the question to a vector
        question_embedding = embedding_model.encode(question).tolist()
        
        # B. Search the database for the 3 closest matches
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        
        # Extract the retrieved text chunks
        retrieved_chunks = results['documents'][0]
        context = "\n\n".join(retrieved_chunks)
        
        # C. Construct the prompt for the LLM
        prompt = f"""
        Answer the user's question using ONLY the following context.
        If the answer is not in the context, say "I don't know based on the provided context."
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
        
        # D. Get the answer from the LLM
        print("[System] Generating answer using Gemini...")
        response = llm.invoke(prompt)
        print(f"\n🤖 AI: {response.content}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
