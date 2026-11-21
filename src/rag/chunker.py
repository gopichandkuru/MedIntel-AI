
def split_text_into_chunks(text, chunk_size=1000, overlap=150):
    """
    Split text into smaller chunks.

    chunk_size: Maximum approximate number of characters.
    overlap: Characters repeated between consecutive chunks.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Try to end at a nearby word boundary
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap

        # Avoid getting stuck
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_medical_documents(documents, chunk_size=1000, overlap=150):
    """
    Split cleaned medical documents into chunks.

    Preserve document name, page number and extraction method.
    """

    all_chunks = []

    for document in documents:

        for page in document["pages"]:

            text = page["text"]

            page_chunks = split_text_into_chunks(
                text,
                chunk_size=chunk_size,
                overlap=overlap
            )

            for chunk_index, chunk in enumerate(page_chunks):

                all_chunks.append({
                    "document_name": document["document_name"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "page_text": text,
                    "extraction_method": page["extraction_method"],
                    "needs_review": (
                        page["extraction_method"] != "native"
                    )
                })

    return all_chunks
