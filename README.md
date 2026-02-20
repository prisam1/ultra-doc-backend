 # UltraDoc Intelligence: TMS Logistics POC 

 UltraDoc is an AI-powered document auditor designed for Transportation Management Systems (TMS). It allows users to upload logistics documents (PDF/DOCX) and perform grounded Q&A and structured data extraction with built-in hallucination guardrails.
 
  # Submission Links
  # Hosted UI: https://ultra-doc-frontend.vercel.app/

  # GitHub Repository:  https://github.com/prisam1/ultra-doc-backend.git 
 
  # Architecture
  The system utilizes a Modular RAG (Retrieval-Augmented Generation) architecture:
  Backend: FastAPI for high-performance, asynchronous API handling.
  Vector Database: FAISS (Facebook AI Similarity Search) for high-speed semantic retrieval.
  LLM: Gemini 2.5 Flash via LangChain for grounded reasoning and extraction.
  Frontend: React with a modern, full-viewport dashboard layout.
  
  # Chunking Strategy
  We employ a Recursive Character Text Splitting strategy:
  Chunk Size: 1,000 characters.
  Overlap: 100 characters.
  Reasoning: Logistics documents contain dense tables where context (like a Rate) can be separated from its label. Overlapping ensures that key-value pairs aren't lost if they fall on a split boundary.

  # Retrieval Method
  The system uses Similarity Search with Scores ($k=5$):
  Instead of looking at one part of the document, the system retrieves the 5 most relevant chunks.
  This is critical for logistics documents where a "Carrier Name" may appear in the header, body, and signature. By retrieving multiple chunks, the LLM can reconcile redundant information for higher accuracy.
  
  # Guardrails & Confidence Scoring
  Hallucination Guardrails
  The system implements a multi-layer guardrail approach:
  Context Strictness: The LLM is system-prompted to only use provided context.
  Fallback String: If the context is missing the specific data point, the system is programmed to return: "Not found in document."
  Heuristic Override: If the mathematical confidence score is below 0.35, the system automatically triggers the "Not Found" state to prevent guessing.
  # Confidence Scoring Logic
  Confidence is calculated using a Heuristic Similarity Score derived from the vector space:
 
  $$Confidence = 1 - (\text{FAISS L2 Distance} / 1.5)$$
  
  The raw distance is normalized into a percentage. 
  A score of 60-70% represents a strong semantic match in complex documents, while lower scores signal potential ambiguity.
  < 35%: Triggers the hallucination guardrail.

  # Failure Cases & Improvements
Failure Case: Non-selectable (Scanned/Image-only) PDFs.

Improvement: Integrate OCR (Tesseract/Google Vision) for image-based documents.

Failure Case: Complex nested tables across page breaks.

Improvement: Implement Layout-Aware Parsing (e.g., Unstructured.io) to maintain table row integrity.

Improvement Idea: Add Agentic Retrieval where the AI can decide to re-scan the document with different parameters if the first retrieval confidence is low.

# Local Setup & Installation

Install dependencies: pip install -r requirements.txt

Environment Variables: Create a .env file and add:
GOOGLE_API_KEY=api_key_here

Run Server: python main.py