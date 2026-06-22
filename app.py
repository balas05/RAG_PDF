import os
from contextlib import asynccontextmanager
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import uuid

# Load API key from .env file
load_dotenv()

# Global variables to store our models
embedding_model = None
chroma_client = None
collection = None
cache_collection = None
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, chroma_client, collection, cache_collection, llm
    print("Initializing RAG pipeline...")
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="my_documents")
    
    # Create or get the semantic cache collection
    cache_collection = chroma_client.get_or_create_collection(name="semantic_cache")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY is missing! Please add it to a .env file.")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    print("RAG initialized.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/query")
def query(request_data: dict):
    question = request_data.get("question")
    
    if not question:
        return JSONResponse(status_code=400, content={"error": "No question provided"})
        
    if not llm:
        return JSONResponse(status_code=500, content={"error": "LLM not initialized. Check API Key."})

    try:
        # 1. Convert the exact question to a vector for semantic caching
        pure_question_embedding = embedding_model.encode(question).tolist()
        
        # 2. Check the Semantic Cache
        cache_results = cache_collection.query(
            query_embeddings=[pure_question_embedding],
            n_results=1
        )
        
        # If we have a match and the distance is below 0.15 (very similar)
        if cache_results['distances'] and len(cache_results['distances'][0]) > 0:
            distance = cache_results['distances'][0][0]
            if distance < 0.18:
                print(f"🔥 Semantic Cache Hit! (Distance: {distance})")
                cached_data = cache_results['metadatas'][0][0]
                return {
                    "answer": cached_data['answer'], 
                    "context": cached_data['context']
                }

        print("No cache match found. Generating new answer from LLM...")

        # 3. Preprocess and expand query for better database retrieval
        query_text = question.lower()
        if "budget" in query_text:
            query_text += " allocation expenditure funding amount capital"
            
        expanded_query = f"""
        Represent the question for retrieving supporting documents:
        {query_text}
        """
        
        # Convert expanded question to vector for document retrieval
        retrieval_embedding = embedding_model.encode(expanded_query).tolist()
        
        # Search the database for 20 chunks
        results = collection.query(
            query_embeddings=[retrieval_embedding],
            n_results=20
        )
        
        # Extract context
        if not results['documents'] or len(results['documents'][0]) == 0:
            context = "No relevant documents found."
        else:
            retrieved_chunks = results['documents'][0]
            context = "\n\n".join(retrieved_chunks)
            
            # Debug: print top retrieved chunks to console
            print("\n===== RETRIEVED CHUNKS =====")
            for i, chunk in enumerate(retrieved_chunks[:3]):
                print(f"\nChunk {i+1}:\n{chunk[:200]}...")
            print("============================")
        
        # Construct a more balanced prompt
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
        
        # Get answer via Gemini
        response = llm.invoke(prompt)
        
        # Prepare result
        result = {"answer": response.content, "context": context}
        
        # 4. Save the brand new question and answer to the Semantic Cache
        cache_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[pure_question_embedding],
            documents=[question],
            metadatas=[{
                "answer": response.content, 
                "context": context
            }]
        )
        
        return result
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    print("Starting FastAPI app...")
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
