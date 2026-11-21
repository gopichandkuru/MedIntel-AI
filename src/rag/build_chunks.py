
import json
from pathlib import Path

from src.rag.pdf_loader import load_medical_documents
from src.rag.text_cleaner import clean_medical_documents
from src.rag.chunker import chunk_medical_documents


def save_chunks(chunks, output_path):
    """
    Save chunks as JSONL.

    Each line contains one chunk and its metadata.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:

        for index, chunk in enumerate(chunks):

            chunk_record = {
                "chunk_id": (
                    f"{chunk['document_name']}"
                    f"_page_{chunk['page_number']}"
                    f"_chunk_{chunk['chunk_index']}"
                ),
                **chunk
            }

            file.write(
                json.dumps(chunk_record, ensure_ascii=False) + "\n"
            )

    print(f"\nChunks saved to: {output_path}")
    print(f"Total chunks saved: {len(chunks)}")


def main():

    # Step 1: Load PDF documents
    documents = load_medical_documents()

    # Step 2: Clean extracted text
    cleaned_documents = clean_medical_documents(documents)

    # Step 3: Generate chunks
    chunks = chunk_medical_documents(
        cleaned_documents,
        chunk_size=1000,
        overlap=150
    )

    # Step 4: Save chunks
    save_chunks(
        chunks,
        "data/processed/rag/medical_chunks.jsonl"
    )

    # Step 5: Display summary
    review_count = sum(
        chunk["needs_review"] for chunk in chunks
    )

    print("\nRAG Chunk Summary")
    print("-----------------")
    print("Total chunks:", len(chunks))
    print("Chunks needing review:", review_count)
    print("Chunks not flagged for review:", len(chunks) - review_count)


if __name__ == "__main__":
    main()
