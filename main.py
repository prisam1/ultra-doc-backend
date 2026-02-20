import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure this is set in your environment
API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# State Management
state = {"vector_db": None}

class QuestionRequest(BaseModel):
    question: str

# Factories
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Or gemini-2.5-flash based on your access
        google_api_key=API_KEY,
        version="v1",
        temperature=0,
        max_retries=10
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY,
        task_type="retrieval_document"
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        state["vector_db"] = FAISS.from_documents(chunks, get_embeddings())
        
        os.remove(temp_path)
        return {"status": "success", "message": f"{file.filename} processed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract")
async def extract_data():
    if not state["vector_db"]:
        raise HTTPException(status_code=400, detail="Upload a document first.")
    try:
        docs = state["vector_db"].docstore._dict.values()
        context = " ".join([d.page_content for d in docs])

        prompt = f"""
        You are a Logistics Data Entry Specialist. 
        Scan the text below for shipment details. 
        NOTE: Data like 'Carrier Name' may appear multiple times (e.g., in a header and a signature). 
        Extract the most complete official name.

        Fields to extract: shipment_id, shipper, consignee, pickup_datetime, delivery_datetime, equipment_type, mode, rate, currency, weight, carrier_name.

        Return ONLY raw JSON. Use null if not found.
        
        Text: {context}
        """
        response = get_llm().invoke(prompt)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not state["vector_db"]:
        raise HTTPException(status_code=400, detail="Upload first.")
    try:
        docs_and_scores = state["vector_db"].similarity_search_with_score(request.question, k=5)
 
        best_score = docs_and_scores[0][1]
        # Ensure this becomes a native Python float immediately
        retrieval_conf = float(max(0, min(1, 1 - (best_score / 1.5)))) 

        context_text = "\n---\n".join([d[0].page_content for d in docs_and_scores])
        source_text = docs_and_scores[0][0].page_content[:400]

        template = f"""
        You are a TMS Audit Assistant. Use ONLY the context below.
        If the answer is not explicitly in the context, say exactly: "Not found in document."
        
        Context: {context_text}
        Question: {request.question}
        
        Answer professionally.
        """
         
        response = get_llm().invoke(template)
        answer_text = response.content.strip()

        if "Not found in document" in answer_text:
            return {
                "answer": "Not found in document.", 
                "source": "N/A", 
                "confidence": 0.0
            }

        # Ensure final_confidence is also a native float
        final_confidence = float(retrieval_conf if retrieval_conf > 0.4 else 0.2)

        return {
            "answer": answer_text,
            "source": source_text,
            "confidence": round(final_confidence, 2) 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)