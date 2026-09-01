from fastapi import FastAPI
from pydantic import BaseModel
from src.generate import generate_answer

# FastAPI app
app = FastAPI(
    title="DocuMind RAG API",
    description="Ask questions about your documents and get grounded answers with citations.",
    version="1.0.0"
)

# This defines the shape of the incoming request
class QuestionRequest(BaseModel):
    question: str

# This defines the shape of the outgoing response
class AnswerResponse(BaseModel):
    question: str
    answer: str

# The main endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Send a POST request with a question, get an answer back.
    Example:
    {
        "question": "What is the study plan?"
    }
    """
    answer = generate_answer(request.question)
    return {"question": request.question, "answer": answer}

# A simple home page
@app.get("/")
async def home():
    return {
        "message": "DocuMind RAG API is running!",
        "endpoints": {
            "/ask": "POST a question, get an answer (try it in /docs!)",
            "/docs": "Interactive API documentation"
        }
    }