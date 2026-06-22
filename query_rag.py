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
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
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
    print("RAG System Ready! Type 'exit' to quit.")
    print("="*50 + "\n")
    
    while True:
        question = input("You: ")
        if question.lower() in ['exit', 'quit']:
            break
            
        print("\n[System] Searching database for relevant info...")
        # Preprocess and expand query for better retrieval
        query = question.lower()
        if "budget" in query:
            query += " allocation expenditure funding amount capital"
            
        expanded_query = f"""
        Represent the question for retrieving supporting documents:
        {query}
        """
        
        # A. Convert the question to a vector
        question_embedding = embedding_model.encode(expanded_query).tolist()
        
        # B. Search the database for 20 chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=20
        )
        
        # Extract the retrieved text chunks
        retrieved_chunks = results['documents'][0]
        context = "\n\n".join(retrieved_chunks)
        
        print("\n===== TOP RETRIEVED CHUNKS =====")
        for i, chunk in enumerate(retrieved_chunks[:3]):
            print(f"\nChunk {i+1}:\n{chunk[:200]}...")
        print("================================")
        
        # C. Construct a more balanced prompt for the LLM
        prompt = f"""
        You are a document QA assistant.
        Use the provided context to answer the question.
        If the answer appears partially in multiple chunks, combine the information.
        If exact numbers are unavailable, provide the closest relevant information from the context.
        If the context is completely irrelevant, say "I don't know based on the provided documents."
        
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
