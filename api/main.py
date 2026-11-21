from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag import unified_medical_rag as rag

app = FastAPI(
    title="MedIntel-AI API",
    description="Evidence-Grounded Medical Question Answering System Using LLMs and RAG",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_state = {
    "stores": None,
    "model": None,
    "ready": False,
}


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.on_event("startup")
def load_medical_rag():
    print("Loading unified MedIntel-AI medical RAG...")

    stores = rag.load_stores()
    model = rag.SentenceTransformer(rag.EMBED_MODEL)

    rag_state["stores"] = stores
    rag_state["model"] = model
    rag_state["ready"] = True

    for store in stores:
        print(f"{store['name']}: {len(store['records'])} records")

    print("Unified medical RAG loaded successfully.")


@app.get("/")
def root():
    return {
        "service": "MedIntel-AI",
        "status": "running",
        "version": "2.0.0",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy" if rag_state["ready"] else "loading",
        "service": "MedIntel-AI",
        "rag_ready": rag_state["ready"],
        "stores": (
            [store["name"] for store in rag_state["stores"]]
            if rag_state["stores"] else []
        ),
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not rag_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Medical RAG pipeline is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = rag.answer_question(
            question,
            rag_state["model"],
            rag_state["stores"],
        )
        return result

    except Exception as error:
        print("Medical answer generation failed:", repr(error))
        raise HTTPException(
            status_code=502,
            detail="The medical answer could not be generated. Please try again.",
        )
