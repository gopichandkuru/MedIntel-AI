
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# PROJECT PATHS AND CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = ROOT / "data/processed/medquad/retrieval"

EMBEDDINGS_FILE = INDEX_DIR / "medquad_embeddings.npy"
METADATA_FILE = INDEX_DIR / "medquad_metadata.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5

# Temporary threshold for inspection.
# This is NOT a medically validated cutoff.
DEFAULT_MIN_SCORE = 0.45


# ==========================================
# MEDQUAD SEMANTIC SEARCH CLASS
# ==========================================

class MedQuADSearch:

    def __init__(self):

        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {EMBEDDINGS_FILE}"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}"
            )

        # Load embeddings
        print("Loading MedQuAD embeddings...")

        self.embeddings = np.load(EMBEDDINGS_FILE)

        # Load metadata
        print("Loading MedQuAD metadata...")

        with METADATA_FILE.open("r", encoding="utf-8") as file:
            self.metadata = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        # Validate index
        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional array."
            )

        if len(self.metadata) != len(self.embeddings):
            raise ValueError(
                "Embedding and metadata counts do not match."
            )

        if self.embeddings.shape[1] != 384:
            raise ValueError(
                f"Expected 384 embedding dimensions, "
                f"got {self.embeddings.shape[1]}."
            )

        if not np.isfinite(self.embeddings).all():
            raise ValueError(
                "Embeddings contain invalid numeric values."
            )

        # Normalize stored embeddings for cosine similarity
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        if np.any(norms == 0):
            raise ValueError(
                "One or more stored embeddings have zero norm."
            )

        self.embeddings = self.embeddings / norms

        # Load embedding model on CPU
        print("Loading embedding model on CPU...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        print(
            f"MedQuAD search ready: {len(self.metadata)} records"
        )

    # ======================================
    # SEARCH METHOD
    # ======================================

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        min_score=DEFAULT_MIN_SCORE
    ):

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not -1 <= min_score <= 1:
            raise ValueError(
                "min_score must be between -1 and 1."
            )

        # Generate normalized query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # Calculate cosine similarity
        scores = self.embeddings @ query_embedding

        # Get highest scoring records
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            score = float(scores[index])

            record = self.metadata[int(index)]

            # Preserve low-scoring results for inspection,
            # but mark whether they passed the threshold.
            results.append({

                "score": score,

                "passed_threshold": score >= min_score,

                "id": record.get("id"),

                "document_id": record.get("document_id"),

                "question": record.get("question"),

                "answer": record.get("answer"),

                "question_type": record.get("question_type"),

                "focus": record.get("focus"),

                "source": record.get("source"),

                "source_url": record.get("source_url"),

                "license": record.get("license")

            })

        return results


# ==========================================
# DISPLAY SEARCH RESULTS
# ==========================================

def display_results(results, min_score=DEFAULT_MIN_SCORE):

    print("\n========== MEDQUAD SEARCH RESULTS ==========")

    if not results:

        print("No results available.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\nResult {rank}")

        print("-" * 60)

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print(
            f"Threshold passed: "
            f"{'YES' if result['passed_threshold'] else 'NO'}"
        )

        print(
            f"Question: {result['question']}"
        )

        print(
            f"Answer: {result['answer']}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Source URL: {result['source_url']}"
        )

        print(
            f"License: {result['license']}"
        )

        print("-" * 60)

    passed_count = sum(
        result["passed_threshold"]
        for result in results
    )

    print(
        f"\nResults passing temporary threshold "
        f"({min_score:.2f}): {passed_count}/{len(results)}"
    )


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    try:

        search_engine = MedQuADSearch()

        while True:

            query = input(
                "\nEnter a medical question "
                "(or type 'exit' to quit): "
            ).strip()

            if query.lower() in {"exit", "quit"}:
                print("MedQuAD search stopped.")
                break

            if not query:
                print("Please enter a question.")
                continue

            results = search_engine.search(
                query=query,
                top_k=DEFAULT_TOP_K,
                min_score=DEFAULT_MIN_SCORE
            )

            display_results(
                results,
                min_score=DEFAULT_MIN_SCORE
            )

    except FileNotFoundError as error:

        print(f"\nFile Error: {error}")

    except ValueError as error:

        print(f"\nValidation Error: {error}")

    except Exception as error:

        print(f"\nUnexpected Error: {error}")
