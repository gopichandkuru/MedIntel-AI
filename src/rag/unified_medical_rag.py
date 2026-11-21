#!/usr/bin/env python3
"""
Unified Medical RAG pipeline for MedIntel-AI.
Auto-detects drug filters and retrieval depth from the prompt, supports multi-turn
conversation memory via Ollama /api/chat, and grounds answers in DailyMed labels.
"""
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
DEFAULT_LLM_MODEL = os.getenv("MEDINTEL_LLM_MODEL", "qwen2.5:3b")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class UnifiedMedicalRAG:
    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_CHAT_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        device: str = "cpu",
    ):
        self.processed_dir = Path(processed_dir)
        self.ollama_url = ollama_url.replace("/api/generate", "/api/chat")
        self.llm_model = llm_model
        self.embedder = SentenceTransformer(embed_model_name, device=device)

        self.embeddings: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []
        self.metadata: list[dict[str, Any]] = []
        self.available_drugs: list[str] = []
        self._load_stores()

    def _load_stores(self) -> None:
        embedding_arrays = []
        drugs = set()
        for emb_path in sorted(self.processed_dir.rglob("*_embeddings.npy")):
            prefix = emb_path.name.replace("_embeddings.npy", "")
            rec_path = emb_path.with_name(f"{prefix}_records.jsonl")
            meta_path = emb_path.with_name(f"{prefix}_metadata.jsonl")

            if not (rec_path.exists() and meta_path.exists()):
                continue

            embs = np.load(emb_path).astype(np.float32)
            recs = load_jsonl(rec_path)
            metas = load_jsonl(meta_path)

            if len(embs) == len(recs) == len(metas) and len(embs) > 0:
                embedding_arrays.append(embs)
                self.records.extend(recs)
                self.metadata.extend(metas)
                for m in metas:
                    if m.get("drug_name"):
                        drugs.add(m["drug_name"].lower())

        if not embedding_arrays:
            raise FileNotFoundError(
                f"No retrieval stores found under {self.processed_dir}. "
                "Run `python3 src/data/ingest_dailymed.py` first."
            )

        self.embeddings = np.vstack(embedding_arrays)
        self.available_drugs = sorted(drugs)

    def _infer_drug_filter(
        self, query: str, history: list[dict[str, Any]] | None = None
    ) -> str | None:
        q_lower = query.lower()
        mentioned = [d for d in self.available_drugs if d in q_lower]
        if len(mentioned) == 1:
            return mentioned[0]
        if len(mentioned) > 1:
            return None  # Comparing multiple drugs — search all

        # Follow-up question: check the most recent turns for a single drug context
        if history:
            for msg in reversed(history[-4:]):
                text = msg.get("content", "").lower()
                prev_mentioned = [d for d in self.available_drugs if d in text]
                if len(prev_mentioned) == 1:
                    return prev_mentioned[0]
        return None

    def _infer_top_k(self, query: str) -> int:
        q_lower = query.lower()
        broad_terms = ("compare", "difference", "all ", "everything", "interactions", "warnings")
        if any(term in q_lower for term in broad_terms) or len(query.split()) > 12:
            return 6
        return 4

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        drug_filter: str | None = None,
        min_score: float = 0.30,
    ) -> list[dict[str, Any]]:
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)

        scores = self.embeddings @ query_vec
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            meta = self.metadata[idx]
            if drug_filter and meta.get("drug_name", "").lower() != drug_filter.lower():
                continue
            results.append({
                "score": score,
                "text": self.records[idx]["text"],
                "metadata": meta,
            })
            if len(results) >= top_k:
                break
        return results

    def _build_messages(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        drugs_str = ", ".join(self.available_drugs) or "loaded medications"
        system_prompt = (
            f"You are MedIntel-AI, a helpful educational medical evidence assistant "
            f"with FDA DailyMed label data for: {drugs_str}. "
            "Use the conversation history to understand follow-up questions. "
            "When medication label evidence is provided, answer using ONLY that evidence "
            "and cite sources inline as [1], [2], etc. "
            "For casual greetings or general chat, respond naturally and concisely. "
            "Never provide personal medical advice or invent label facts."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history[-8:]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        if contexts:
            evidence_blocks = []
            for i, item in enumerate(contexts, start=1):
                meta = item["metadata"]
                header = (
                    f"[{i}] Source: {meta.get('source', 'Unknown')} | "
                    f"Drug: {meta.get('drug_name', 'N/A')} | "
                    f"Section: {meta.get('section', 'N/A')}"
                )
                evidence_blocks.append(f"{header}\n{item['text']}")
            user_content = (
                f"Relevant DailyMed Evidence:\n"
                f"{chr(10).join(evidence_blocks)}\n\n"
                f"User Question: {query}"
            )
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})
        return messages

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        drug_filter: str | None = None,
        use_llm: bool = True,
        min_score: float = 0.30,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        resolved_drug = drug_filter or self._infer_drug_filter(query, history)
        resolved_k = top_k or self._infer_top_k(query)

        # Enrich short follow-ups with the active drug name for better vector recall
        retrieval_query = query
        if resolved_drug and resolved_drug not in query.lower():
            retrieval_query = f"{resolved_drug} {query}"

        contexts = self.retrieve(
            retrieval_query,
            top_k=resolved_k,
            drug_filter=resolved_drug,
            min_score=min_score,
        )

        if not use_llm:
            fallback = contexts[0]["text"] if contexts else "No matching medication evidence found."
            return {
                "query": query,
                "answer": fallback,
                "sources": contexts,
                "llm_used": False,
            }

        messages = self._build_messages(query, contexts, history)
        try:
            resp = requests.post(
                self.ollama_url,
                json={"model": self.llm_model, "messages": messages, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            answer_text = resp.json().get("message", {}).get("content", "").strip()
            return {
                "query": query,
                "answer": answer_text,
                "sources": contexts,
                "llm_used": True,
            }
        except requests.RequestException as exc:
            fallback = contexts[0]["text"] if contexts else "No local evidence matched."
            return {
                "query": query,
                "answer": f"[Local LLM unavailable: {exc}]\n\n{fallback}",
                "sources": contexts,
                "llm_used": False,
            }
