
import io
import re
from contextlib import redirect_stdout

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag import generate_answer as rag


app = FastAPI(
    title="MedIntel-AI API",
    description="Educational, source-grounded medical knowledge assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Loaded once when the API starts
rag_state = {
    "embeddings": None,
    "documents": None,
    "model": None,
    "ready": False,
}


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)




# ==================================================
# MEDICAL DOMAIN RESTRICTION
# ==================================================

MEDICAL_DOMAIN_CUES = (
    "medical", "medicine", "health", "healthcare",
    "patient", "doctor", "physician", "hospital", "clinic",
    "disease", "disorder", "illness", "symptom", "symptoms",
    "diagnosis", "diagnose", "treatment", "therapy",
    "medication", "medicine", "drug", "dosage", "dose",
    "prescription", "side effect", "adverse effect",
    "vaccine", "vaccination", "immunization",
    "infection", "virus", "bacteria", "antibiotic",
    "diabetes", "diabetic", "hypoglycemia",
    "blood glucose", "glucose", "insulin", "hba1c",
    "blood pressure", "hypertension", "heart disease",
    "cardiac", "stroke", "kidney", "renal", "liver",
    "hepatic", "lung", "respiratory", "asthma", "copd",
    "cancer", "tumor", "tumour", "pregnancy", "pregnant",
    "nutrition", "obesity", "fever", "cough", "headache",
    "migraine", "pain", "nausea", "vomiting",
    "diarrhea", "diarrhoea", "rash", "wound", "bleeding",
    "fracture", "bone", "joint", "muscle", "skin",
    "allergy", "immune", "mental health", "depression",
    "anxiety", "neurology", "cardiology", "dermatology",
    "pediatrics", "paediatrics", "blood test", "urine test",
    "lab result", "blood count", "hemoglobin",
    "haemoglobin", "cholesterol", "thyroid",
    "metformin", "paracetamol", "acetaminophen",
    "aspirin", "ibuprofen", "amoxicillin", "amlodipine",
    "atorvastatin", "emergency", "first aid"
)

def is_medical_question(question):
    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        question.lower()
    ).strip()

    # Reject programming and software-development requests.
    coding_request = re.search(
        r"\b(write|generate|debug|fix|implement|build|create)\b"
        r".{0,80}\b(code|script|program|software|website|web app|algorithm)\b",
        normalized
    )

    if coding_request:
        return False

    return any(
        re.search(
            rf"\b{re.escape(cue)}\b",
            normalized
        )
        for cue in MEDICAL_DOMAIN_CUES
    )


@app.on_event("startup")
def load_medical_rag():
    print("Loading MedIntel-AI WHO RAG pipeline...")

    embeddings, documents = rag.load_rag_data()
    model = rag.load_embedding_model()

    rag_state["embeddings"] = embeddings
    rag_state["documents"] = documents
    rag_state["model"] = model
    rag_state["ready"] = True

    print("WHO RAG pipeline loaded successfully.")


@app.get("/")
def root():
    return {
        "message": "Welcome to MedIntel-AI API",
        "status": "running",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "MedIntel-AI",
        "rag_ready": rag_state["ready"],
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

    # Reject questions outside the medical domain.
    if not is_medical_question(question):
        return {
            "answer": (
                "I'm MedIntel-AI, a medical-domain assistant. "
                "I can only help with medical and healthcare-related "
                "questions. Please ask about a health condition, "
                "symptom, medicine, test, treatment or healthcare topic."
            ),
            "sources": [],
            "status": "out_of_domain",
            "warnings": [
                "This question was rejected before medical document retrieval."
            ],
        }

    try:
        retrieved_chunks = rag.retrieve_chunks(
            question,
            rag_state["model"],
            rag_state["embeddings"],
            rag_state["documents"],
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "The provided WHO document does not contain "
                    "enough verified information to answer this question."
                ),
                "sources": [],
                "status": "abstained",
                "warnings": [
                    "No suitable source passages were retrieved.",
                    "This assistant does not provide diagnosis or prescriptions.",
                ],
            }

        # Convert retrieved document metadata into JSON
        sources = []

        for chunk in retrieved_chunks:
            source = {
                "document": chunk.get(
                    "document_name", "WHO medical document"
                ),
                "page": chunk.get("page_number"),
                "extraction_method": chunk.get(
                    "extraction_method", "Unknown"
                ),
                "similarity": round(
                    float(chunk.get("similarity", 0.0)), 4
                ),
            }

            if source not in sources:
                sources.append(source)

        # Assign stable IDs to the deduplicated source records.
        for index, source in enumerate(sources, 1):
            source["id"] = f"S{index}"

        warnings = [
            "Educational document-based information only. "
            "Not a diagnosis or medical advice."
        ]

        if any(
            "ocr" in str(
                chunk.get("extraction_method", "")
            ).lower()
            for chunk in retrieved_chunks
        ):
            warnings.append(
                "Some retrieved text was OCR-extracted. "
                "Verify the original PDF before relying on it."
            )

        # 1. Referral criteria: preserve existing extraction
        if re.search(
            r"\b(referr\w*|refer\w*|higher healthcare|tertiary care)\b",
            question,
            re.IGNORECASE,
        ):
            evidence = rag.extract_referral_criteria(
                retrieved_chunks
            )

            if not evidence:
                return {
                    "answer": (
                        "Referral criteria could not be reliably "
                        "extracted from the retrieved passages. "
                        "Please verify the original document."
                    ),
                    "sources": sources,
                    "status": "abstained",
                    "warnings": warnings,
                }

            answer = (
                "Referral criteria listed in the document:\n\n"
                + "\n".join(
                    f"{index}. {item}"
                    for index, item in enumerate(
                        evidence["criteria"], 1
                    )
                )
                + f"\n\nSource page: {evidence['page']}"
            )

            return {
                "answer": answer,
                "sources": sources,
                "status": "answered",
                "warnings": warnings,
            }

        # 2. Review and follow-up: preserve existing extraction
        if re.search(
            r"\b(review|reviewing|follow[- ]?up|monitoring)\b",
            question,
            re.IGNORECASE,
        ):
            evidence = rag.extract_review_evidence(
                retrieved_chunks
            )

            if not evidence:
                return {
                    "answer": (
                        "The retrieved passages did not contain "
                        "clear review-related statements."
                    ),
                    "sources": sources,
                    "status": "abstained",
                    "warnings": warnings,
                }

            answer = (
                "Review and follow-up statements extracted "
                "from the document:\n\n"
                + "\n".join(
                    f"{index}. {item['text']} "
                    f"(Source page: {item['page']})"
                    for index, item in enumerate(evidence, 1)
                )
            )

            return {
                "answer": answer,
                "sources": sources,
                "status": "answered",
                "warnings": warnings,
            }

        # 3. Other questions: use existing WHO + Ollama generation
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            rag.generate_llm_answer(
                question,
                retrieved_chunks,
            )

        answer = captured_output.getvalue().strip()

        if not answer:
            return {
                "answer": (
                    "The system could not generate an answer "
                    "from the retrieved passages."
                ),
                "sources": sources,
                "status": "abstained",
                "warnings": warnings,
            }

        if "Could not connect to Ollama" in answer:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Could not connect to Ollama. "
                    "Please check whether ollama serve is running."
                ),
            )

        return {
            "answer": answer,
            "sources": sources,
            "status": "answered",
            "warnings": warnings,
        }

    except HTTPException:
        raise

    except Exception as error:
        print("Chat API error:", error)

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the question.",
        )
