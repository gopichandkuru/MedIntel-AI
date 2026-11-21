
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# 1. File paths
INPUT_FILE = Path("data/processed/rag/medical_chunks.jsonl")
OUTPUT_DIR = Path("data/processed/rag")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    # 2. Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Chunk file not found: {INPUT_FILE}")

    # 3. Load chunks
    chunks = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("No chunks found.")

    texts = [chunk["text"] for chunk in chunks]

    print("Total chunks:", len(texts))

    # 4. Select GPU when available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 5. Load embedding model
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Embedding model loaded.")
    print("Embedding dimensions:", model.get_embedding_dimension())

    # 6. Convert text into embeddings
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 7. Validate embeddings
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    print("Embedding shape:", embeddings.shape)

    # 8. Save embedding vectors
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        OUTPUT_DIR / "medical_embeddings.npy",
        embeddings
    )

    # 9. Save matching chunk metadata
    metadata_file = OUTPUT_DIR / "embedding_metadata.jsonl"

    with metadata_file.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\nEmbeddings created successfully!")
    print("Saved vectors:", OUTPUT_DIR / "medical_embeddings.npy")
    print("Saved metadata:", metadata_file)
    print("Total vectors:", len(embeddings))


if __name__ == "__main__":
    main()
