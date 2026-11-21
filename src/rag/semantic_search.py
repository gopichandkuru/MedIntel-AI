
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# File paths
EMBEDDINGS_FILE = Path(
    "data/processed/rag/medical_embeddings.npy"
)

METADATA_FILE = Path(
    "data/processed/rag/embedding_metadata.jsonl"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    # 1. Load stored embeddings
    embeddings = np.load(EMBEDDINGS_FILE)

    # 2. Load matching metadata
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file]

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts do not match.")

    # 3. Load the same embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Semantic Search Ready!")
    print("Documents:", len(metadata))

    # 4. Get a question from the user
    query = input("\nEnter your search question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    # 5. Convert question into an embedding
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


    # 6. Calculate cosine similarity
    scores = np.dot(embeddings, query_embedding)

    # 7. Apply state-specific filtering
    protocol_page_map = {
        "karnataka": {14},
        "madhya pradesh": {16},
        "maharashtra": {17},
        "meghalaya": {18},
        "telangana": {22},
        "uttarakhand": {24},
        "uttar pradesh": {25},
        "goa": {9, 11}
    }

    question_lower = query.lower()
    selected_state = None

    for state in protocol_page_map:
        if state in question_lower:
            selected_state = state
            break

    candidate_indices = np.arange(len(metadata))

    if selected_state:
        allowed_pages = protocol_page_map[selected_state]

        candidate_indices = np.array([
            i for i, chunk in enumerate(metadata)
            if chunk["page_number"] in allowed_pages
        ])

        print(f"\nState filter applied: {selected_state.title()}")

    if len(candidate_indices) == 0:
        print("No matching state documents found.")
        return

    filtered_scores = scores[candidate_indices]

    top_k = min(5, len(candidate_indices))

    local_indices = np.argsort(filtered_scores)[::-1][:top_k]
    top_indices = candidate_indices[local_indices]

    # 8. Display results
    print("\nTop matching document chunks")
    print("=" * 60)

    for rank, index in enumerate(top_indices, start=1):

        chunk = metadata[index]

        print(f"\nResult {rank}")
        print("Chunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page_number"])
        print("Similarity:", round(float(scores[index]), 4))
        print("Extraction:", chunk["extraction_method"])
        print("Needs human review:", chunk["needs_review"])
        print("\nText:")
        print(chunk["text"])
        print("-" * 60)


if __name__ == "__main__":
    main()
