#!/usr/bin/env python3
"""
FastAPI backend for MedIntel-AI.
Exposes health checks and the Unified Medical RAG pipeline.
"""
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.unified_medical_rag import UnifiedMedicalRAG

rag_pipeline: UnifiedMedicalRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    rag_pipeline = UnifiedMedicalRAG(
        llm_model=os.getenv("MEDINTEL_LLM_MODEL", "qwen2.5:3b")
    )
    yield
    rag_pipeline = None


app = FastAPI(
    title="MedIntel-AI API",
    description="Educational Medical Machine Learning & Evidence Assistant API.",
    version="0.1.0",
    lifespan=lifespan,
)


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Medical question to answer.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of evidence chunks.")
    drug_filter: str | None = Field(
        default=None, description="Optional drug name filter (e.g., metformin, glimepiride)."
    )
    use_llm: bool = Field(
        default=True, description="Generate a synthesized answer using local Ollama LLM."
    )


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    llm_used: bool
    sources: list[dict[str, Any]]


@app.get("/health")
def health_check():
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    return {
        "status": "ok",
        "chunks_loaded": len(rag_pipeline.records),
        "llm_model": rag_pipeline.llm_model,
    }


@app.post("/rag/query", response_model=RAGQueryResponse)
def query_rag(payload: RAGQueryRequest):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    return rag_pipeline.answer(
        query=payload.query,
        top_k=payload.top_k,
        drug_filter=payload.drug_filter,
        use_llm=payload.use_llm,
    )


@app.post("/rag/retrieve")
def retrieve_evidence(payload: RAGQueryRequest):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    results = rag_pipeline.retrieve(
        query=payload.query,
        top_k=payload.top_k,
        drug_filter=payload.drug_filter,
    )
    return {"query": payload.query, "results": results}
