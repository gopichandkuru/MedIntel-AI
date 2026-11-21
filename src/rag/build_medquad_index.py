
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Project paths
ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/medquad_v4/direct_answers/train.jsonl"
OUTPUT_DIR = ROOT / "data/processed/medquad_v4/retrieval"

EMBEDDINGS_FILE = OUTPUT_DIR / "medquad_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "medquad_metadata.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_medquad_records():
    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            question = record.get("question", "").strip()
            answer = record.get("answer", "").strip()

            if not question or not answer:
                continue

            records.append(record)

    if not records:
        raise ValueError("No valid MedQuAD records found.")

    return records


def build_search_text(record):
    question = record.get("question", "").strip()
    focus = record.get("focus", "")

    if isinstance(focus, list):
        focus = " ".join(str(item) for item in focus)
    else:
        focus = str(focus)

    return f"Question: {question}. Medical topic: {focus}".strip()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MedQuAD training records...")
    records = load_medquad_records()
    print(f"Loaded {len(records)} records.")

    print("\nLoading embedding model on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    search_texts = [build_search_text(record) for record in records]

    print("\nGenerating embeddings...")
    embeddings = model.encode(
        search_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print("\nSaving embeddings...")
    np.save(EMBEDDINGS_FILE, embeddings)

    print("Saving source metadata...")
    with METADATA_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            metadata = {
                "record_id": record.get("record_id"),
                "source_document_id": record.get("source_document_id"),
                "xml_filename": record.get("xml_filename"),
                "id": record.get("id"),
                "document_id": record.get("document_id"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "question_type": record.get("question_type"),
                "focus": record.get("focus"),
                "source": record.get("source"),
                "source_url": record.get("source_url"),
                "license": record.get("license"),
            }

            file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print("\nMedQuAD retrieval index created successfully!")
    print(f"Records: {len(records)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings: {EMBEDDINGS_FILE}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
