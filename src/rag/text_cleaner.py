
import re


def clean_text(text):
    """
    Clean extracted PDF text without changing its meaning.

    - Remove form-feed characters.
    - Normalize spaces and tabs.
    - Remove unnecessary blank lines.
    - Preserve meaningful line breaks.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Fix common PDF ligature and encoding artifacts
    replacements = {
        "Ɵ": "ti",
        "ƞ": "tf",
        "Ō": "ft",
        "ﬁ": "fi",
        "ﬂ": "fl"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove form-feed characters
    text = text.replace("\x0c", "\n")

    # Normalize spaces and tabs on each line
    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()

        if line:
            lines.append(line)

    # Join lines while preserving their order
    cleaned_text = "\n".join(lines)

    return cleaned_text


def clean_medical_documents(documents):
    """
    Clean extracted pages while preserving source metadata.

    Original text is retained as raw_text.
    """

    cleaned_documents = []

    for document in documents:

        cleaned_pages = []

        for page in document["pages"]:

            raw_text = page["text"]
            cleaned = clean_text(raw_text)

            cleaned_pages.append({
                "page_number": page["page_number"],
                "raw_text": raw_text,
                "text": cleaned,
                "extraction_method": page["extraction_method"]
            })

        cleaned_documents.append({
            "document_name": document["document_name"],
            "pages": cleaned_pages
        })

    return cleaned_documents
