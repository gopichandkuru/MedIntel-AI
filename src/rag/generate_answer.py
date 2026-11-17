
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "qwen3.5:2b-q4_K_M"

TOP_K = 8

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    PROJECT_ROOT / "data/processed/rag/medical_embeddings.npy"
)

METADATA_FILE = (
    PROJECT_ROOT / "data/processed/rag/embedding_metadata.jsonl"
)

CHUNKS_FILE = (
    PROJECT_ROOT / "data/processed/rag/medical_chunks.jsonl"
)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


# ==================================================
# LOAD DATA
# ==================================================

def load_jsonl(file_path):

    records = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


def load_rag_data():

    required_files = [
        EMBEDDINGS_FILE,
        METADATA_FILE,
        CHUNKS_FILE
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    embeddings = np.load(EMBEDDINGS_FILE)

    metadata = load_jsonl(METADATA_FILE)
    chunks = load_jsonl(CHUNKS_FILE)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    if not (
        len(embeddings) == len(metadata) == len(chunks)
    ):
        raise ValueError(
            "Embedding, metadata and chunk counts do not match. "
            "Please regenerate embeddings."
        )

    documents = []

    for index, (meta, chunk) in enumerate(
        zip(metadata, chunks)
    ):

        meta_id = meta.get("chunk_id")
        chunk_id = chunk.get("chunk_id")

        if meta_id and chunk_id and meta_id != chunk_id:
            raise ValueError(
                f"Chunk order mismatch at index {index}."
            )

        text = (
            chunk.get("text")
            or chunk.get("chunk_text")
            or chunk.get("content")
            or meta.get("text")
        )

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Missing text at chunk index {index}. "
                "Please check medical_chunks.jsonl."
            )

        # Original chunk data is authoritative.
        item = {**meta, **chunk}
        item["text"] = text.strip()

        documents.append(item)

    if len(documents) != embeddings.shape[0]:
        raise ValueError(
            "Document and embedding alignment failed."
        )

    return embeddings, documents

# ==================================================
# EMBEDDING MODEL
# ==================================================

def load_embedding_model():

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    return model
# ==================================================
# REGIONAL PROTOCOL IDENTIFICATION
# ==================================================

PROTOCOL_PAGE_MAP = {
    "karnataka": {14},
    "madhya pradesh": {16},
    "maharashtra": {17},
    "meghalaya": {18},
    "telangana": {22},
    "uttarakhand": {24},
    "uttar pradesh": {25}
}


def get_requested_protocol_pages(question):

    question = re.sub(
        r"\s+",
        " ",
        question.lower()
    ).strip()

    for region, pages in sorted(
        PROTOCOL_PAGE_MAP.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
 
        if re.search(
            rf"\b{re.escape(region)}\b",
            question
        ):
            return pages

    return None

# ==================================================
# RETRIEVAL
# ==================================================

def retrieve_chunks(question, model, embeddings, documents):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    similarities = np.dot(embeddings, query_embedding)

    requested_pages = get_requested_protocol_pages(question)

    stop_words = {
        "what", "is", "are", "the", "a", "an",
        "and", "or", "of", "in", "on", "to",
        "for", "with", "how", "does", "do",
        "explain", "tell", "me", "about"
    }

    query_words = set(
        re.findall(r"[a-zA-Z]+", question.lower())
    ) - stop_words

    scores = []

    for index, document in enumerate(documents):

        if document.get("needs_review", False):
            continue

        if requested_pages is not None:
            if document.get("page_number") not in requested_pages:
                continue

        text = document["text"].lower()

        document_words = set(
            re.findall(r"[a-zA-Z]+", text)
        )

        overlap = len(query_words & document_words)

        lexical_score = (
            overlap / max(len(query_words), 1)
        )

        combined_score = (
            0.70 * float(similarities[index])
            + 0.30 * lexical_score
        )

        scores.append((combined_score, index))

    scores.sort(reverse=True)

    results = []

    for score, index in scores[:TOP_K]:

        result = documents[index].copy()

        result["similarity"] = float(similarities[index])
        result["combined_score"] = score

        results.append(result)

    return results
# ==================================================
# DISPLAY SOURCES
# ==================================================


def display_sources(retrieved_chunks):

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for index, chunk in enumerate(retrieved_chunks, 1):

        print(f"\nSource {index}")

        print(
            "Document:",
            chunk.get("document_name", "Unknown")
        )

        print(
            "Page:",
            chunk.get("page_number", "Unknown")
        )

        print(
            "Semantic Similarity:",
            round(chunk.get("similarity", 0.0), 4)
        )

        print(
            "Combined Score:",
            round(chunk.get("combined_score", 0.0), 4)
        )

        print(
            "Extraction:",
            chunk.get("extraction_method", "Unknown")
        )

        print(
            "Human review required:",
            chunk.get("needs_review", False)
        )


def display_source_warnings(chunks):

    pages = sorted({
        str(chunk.get("page_number", "Unknown"))
        for chunk in chunks
        if chunk.get("needs_review", False)
    })

    if pages:

        print("\nSOURCE QUALITY WARNING")

        print("OCR-extracted pages requiring verification:", ", ".join(pages))

        print(
            "Check the original PDF before relying on "
            "information from these pages."
        )


# ==================================================
# REFERRAL CRITERIA - DIRECT EXTRACTION
# ==================================================

def extract_referral_criteria(documents):

    heading_pattern = re.compile(
        r"(?im)^\s*4\.8\s+Referral\s+criteria\s+for\s+all\s+visits\s*$"
    )

    stop_pattern = re.compile(
        r"(?im)^\s*(?:\d+\.\d+\s+|Protocol\s+for\s+screening)"
    )

    for chunk in documents:

        text = chunk["text"]

        match = heading_pattern.search(text)

        if not match:
            continue

        section = text[match.end():]

        stop = stop_pattern.search(section)

        if stop:
            section = section[:stop.start()]

        criteria = []
        current_item = None

        for line in section.splitlines():

            line = line.strip()

            if not line:
                continue

            item_match = re.match(
                r"^([a-f])\)\s*(.*)$",
                line,
                flags=re.IGNORECASE
            )

            if item_match:

                if current_item:
                    criteria.append(current_item.strip())

                current_item = item_match.group(2).strip()

            elif current_item:

                current_item += " " + line

        if current_item:
            criteria.append(current_item.strip())

        if criteria:

            return {
                "criteria": criteria,
                "page": chunk.get("page_number", "Unknown"),
                "needs_review": chunk.get("needs_review", False),
                "extraction_method": chunk.get(
                    "extraction_method", "Unknown"
                )
            }

    return None


# ==================================================
# PATIENT REVIEW - DIRECT EVIDENCE EXTRACTION
# ==================================================

def expand_adjacent_chunks(retrieved_chunks, all_documents):

    expanded = {}

    for chunk in retrieved_chunks:

        page = chunk.get("page_number")
        index = chunk.get("chunk_index")

        for doc in all_documents:

            if doc.get("page_number") == page:

                doc_index = doc.get("chunk_index")

                if doc_index is not None and abs(doc_index - index) <= 1:

                    key = doc.get("chunk_id", (page, doc_index))
                    expanded[key] = doc.copy()

    return sorted(
        expanded.values(),
        key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0))
    )



def extract_review_evidence(retrieved_chunks):
    """
    Extract complete review and follow-up statements.
    Join bullet continuations and adjacent chunk text.
    Skip OCR passages requiring human verification.
    """

    evidence = []
    seen = set()

    # Group chunks by page.
    pages = {}

    for chunk in retrieved_chunks:

        if chunk.get("needs_review", False):
            continue

        page = chunk.get("page_number", "Unknown")

        pages.setdefault(page, []).append(chunk)

    for page, chunks in pages.items():

        # Process chunks in their original order.
        chunks.sort(
            key=lambda x: x.get("chunk_index", 0)
        )

        lines = []

        for chunk in chunks:

            chunk_lines = [
                line.strip()
                for line in chunk.get("text", "").splitlines()
                if line.strip()
            ]

            for line in chunk_lines:

                # Avoid repeated overlapping lines.
                if not lines or lines[-1] != line:
                    lines.append(line)

        i = 0

        while i < len(lines):

            line = lines[i]

            # Handle bullet statements and their continuation lines.
            if re.match(r"^[•●▪*-]\s*", line):

                statement_lines = [line]
                j = i + 1

                while j < len(lines):

                    if re.match(r"^[•●▪*-]\s*", lines[j]):
                        break

                    statement_lines.append(lines[j])
                    j += 1

                statement = " ".join(statement_lines)

                if re.search(
                    r"\b(review|follow[- ]?up)\b",
                    statement,
                    re.IGNORECASE
                ):
                    cleaned = re.sub(
                        r"\s+", " ", statement
                    ).strip()

                    # Reject unfinished statements.
                    if not re.search(
                        r"\b(after|in|for|every|and|if|of|to|as|with|increasing|prior)\s*$",
                        cleaned,
                        re.IGNORECASE
                    ) and len(cleaned.split()) >= 4:

                        key = (str(page), cleaned.lower())

                        if key not in seen:

                            seen.add(key)

                            evidence.append({
                                "text": cleaned,
                                "page": page,
                                "needs_review": False,
                                "extraction_method": "native"
                            })

                i = j
                continue

            # Handle standalone review statements.
            if re.search(
                r"\b(review|follow[- ]?up)\b",
                line,
                re.IGNORECASE
            ):

                statement = line

                # Join split time expressions.
                if i + 1 < len(lines):

                    next_line = lines[i + 1]

                    if re.search(
                        r"\b(after|in|for|every)\s+\d+\s*(?:[-–]\s*\d+)?$",
                        statement,
                        re.IGNORECASE
                    ) and re.match(
                        r"^(days?|weeks?|months?|years?)\b",
                        next_line,
                        re.IGNORECASE
                    ):
                        statement += " " + next_line
                        i += 1

                    # Join short continuation after an incomplete phrase.
                    elif re.search(
                        r"\b(after|in|for|every|and|of|to|as|with|increasing|prior)\s*$",
                        statement,
                        re.IGNORECASE
                    ) and len(next_line.split()) <= 6:
                        statement += " " + next_line
                        i += 1

                statement = re.sub(
                    r"\s+", " ", statement
                ).strip()

                # Reject incomplete time expressions.
                incomplete = re.search(
                    r"\b(after|in|for|every)\s+(months?|weeks?|days?|years?)\b",
                    statement,
                    re.IGNORECASE
                )

                unfinished = re.search(
                    r"\b(after|in|for|every|and|if|of|to|as|with)\s*$",
                    statement,
                    re.IGNORECASE
                )

                if (
                    len(statement.split()) >= 3
                    and not incomplete
                    and not unfinished
                ):

                    key = (str(page), statement.lower())

                    if key not in seen:

                        seen.add(key)

                        evidence.append({
                            "text": statement,
                            "page": page,
                            "needs_review": False,
                            "extraction_method": "native"
                        })

            i += 1

    return evidence





def display_review_answer(evidence):

    print("\n" + "=" * 60)
    print("MEDINTEL AI - PATIENT REVIEW INFORMATION")
    print("=" * 60)

    if not evidence:

        print(
            "\nThe retrieved document text did not contain "
            "clear review-related statements."
        )

        return

    print(
        "\nThe following statements were extracted "
        "directly from the document:"
    )

    for index, item in enumerate(evidence, 1):

        print(f"\n{index}. {item['text']}")

        print("   Source page:", item["page"])

        if item["needs_review"]:

            print(
                "   Warning: OCR extraction; "
                "verify the original PDF."
            )

    print("\n" + "-" * 60)

    print(
        "Answer method: Direct source-text extraction."
    )

    print(
        "Note: Separate statements may describe different "
        "treatment stages or monitoring activities."
    )

    print(
        "This output is not a current clinical recommendation."
    )


# ==================================================
# DISPLAY REFERRAL ANSWER
# ==================================================

def display_referral_answer(evidence):

    print("\n" + "=" * 60)
    print("MEDINTEL AI - REFERRAL CRITERIA")
    print("=" * 60)

    print(
        "\nThe document lists the following referral criteria:"
    )

    for index, criterion in enumerate(
        evidence["criteria"], 1
    ):

        print(f"\n{index}. {criterion}")

    print("\nSource page:", evidence["page"])

    print(
        "Answer method: Direct extraction from document."
    )

    if evidence["needs_review"]:

        print(
            "\nWARNING: OCR-extracted passage. "
            "Verify the original PDF."
        )

    print(
        "\nThese are statements from the uploaded document, "
        "not independently verified current clinical guidance."
    )


# ==================================================
# BUILD LLM CONTEXT
# ==================================================


def build_context(chunks):

    parts = []
    seen_pages = set()

    source_index = 1

    for chunk in chunks:

        page_key = (
            chunk.get("document_name"),
            chunk.get("page_number")
        )

        if page_key in seen_pages:
            continue

        seen_pages.add(page_key)

        page_text = (
            chunk.get("page_text")
            or chunk.get("text", "")
        )

        parts.append(
            f"""
SOURCE {source_index}
Page: {chunk.get('page_number')}
Extraction: {chunk.get('extraction_method', 'Unknown')}
Needs human review: {chunk.get('needs_review', False)}

COMPLETE PAGE TEXT:
{page_text}
"""
        )

        source_index += 1

    return "\n".join(parts)



# ==================================================
# GENERAL QUESTIONS - OLLAMA
# ==================================================


def generate_llm_answer(question, chunks):

    context = build_context(chunks)
    question_lower = question.lower()

         # General diabetes symptoms are not supported by this source
    if (
        "symptom" in question_lower
        and "hypoglycemia" not in question_lower
    ):
        print("\nMEDINTEL AI - SOURCE-BASED ANSWER")
        print(
            "\nThe document does not provide a separate, "
            "general list of diabetes symptoms."
        )
        print(
            "It lists symptoms under the hypoglycemia section. "
            "These should not be treated as general diabetes symptoms."
        )
        print("\nSource: who_india_diabetes_protocol.pdf, Page 4")
        return
    # 1. Hypoglycemia symptoms
    if "hypoglycemia" in question_lower and "symptom" in question_lower:

        symptom_terms = [
            "cold",
            "sweat",
            "trembling of hands",
            "hunger",
            "palpitation",
            "confusion",
            "loss of consciousness"
        ]

        for chunk in chunks:
            page_text = chunk.get("page_text") or chunk.get("text", "")
            text_lower = page_text.lower()

            if "hypoglycemia" in text_lower:
                found = [
                    term for term in symptom_terms
                    if term in text_lower
                ]

                if found:
                    print("\nMEDINTEL AI - SOURCE-BASED ANSWER")
                    print("\nSymptoms listed under Hypoglycemia:")

                    for term in found:
                        print("-", term.title())

                    print(
                        f"\nSource: {chunk.get('document_name')}, "
                        f"Page {chunk.get('page_number')}"
                    )

                    print(
                        "\nNote: These are hypoglycemia symptoms, "
                        "not general diabetes symptoms."
                    )
                    return

         # 2. Lifestyle modifications
    if "lifestyle" in question_lower or "life style" in question_lower:

        for chunk in chunks:
            page_text = chunk.get("page_text") or chunk.get("text", "")

            match = re.search(
                r"(?:^|\n)\s*.*?(?=\n\s*Life\s*Style\s*Modification\s*\(LSM\))",
                page_text,
                re.DOTALL
            )

            if match:
                print("\nMEDINTEL AI - SOURCE-BASED ANSWER")
                print("\nLifestyle Modification (LSM):")
                print(match.group(0).strip())

                print(
                    f"\nSource: {chunk.get('document_name')}, "
                    f"Page {chunk.get('page_number')}"
                )

                print("\nNote: Extracted from the provided document.")
                return
    # 3. Regional protocol
    requested_pages = get_requested_protocol_pages(question)

    if requested_pages:
        source = next(
            (
                chunk for chunk in chunks
                if chunk.get("page_number") in requested_pages
            ),
            None
        )

        if source:
            page_text = source.get("page_text") or source.get("text", "")

            print("\nMEDINTEL AI - SOURCE-BASED ANSWER")
            print("\n" + page_text)

            print(
                f"\nSource: {source.get('document_name')}, "
                f"Page {source.get('page_number')}"
            )

            print("\nNote: Extracted source text, not medical advice.")
            return








   
   
    system_prompt = """
You are MedIntel AI, a medical document question-answering assistant.

STRICT GROUNDING RULES:
1. Answer only the exact question asked.
2. Use only information explicitly supported by the supplied passages.
3. Never use outside medical knowledge to fill missing information.
4. Never invent definitions, symptoms, diagnoses, medicines, doses,
   timings, thresholds, treatment steps, or recommendations.
5. Do not combine instructions from different sections into one instruction.
6. Preserve the source's original meaning, terminology, numbers and units.
7. Do not assume that two similar medical terms mean the same thing.
8. Include page citations for every important factual statement.
9. Cite only page numbers present in the supplied passages.
10. If the passages do not answer the question, clearly say:
    "The provided document does not contain enough information to answer this question."

ANSWER FORMAT:
- Give a direct answer first.
- Use short paragraphs or bullet points where appropriate.
- Use headings only when they help answer the question.
- Include only sections relevant to the question.
- Do not force unrelated protocol steps or sections into the answer.
- Do not repeat information.
- If source text is unclear, say so instead of guessing.

SAFETY:
This is document-based information, not a diagnosis or medical advice.
Do not recommend changes to a patient's medication or treatment.
"""

    user_prompt = f"""
QUESTION:
{question}

SOURCE PASSAGES:
{context}

Answer the question using only the source passages above.

Before responding, check:
- Does the source explicitly support the answer?
- Are all numbers, doses and timings copied accurately?
- Are the cited page numbers present in the source passages?
- Have you avoided adding unrelated information?

If the source does not support an answer, say that clearly.
Do not invent missing information.
"""


    payload = {
        "model": LLM_NAME,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "options": {
            "num_ctx": 4096,
            "num_predict": 1400,
            "temperature": 0.0,
            "repeat_penalty": 1.1
        }
    }

    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=request_data,
        headers={"Content-Type": "application/json"}
    )

    start_time = time.perf_counter()

    try:

        with urllib.request.urlopen(
            request,
            timeout=300
        ) as response:

            result = json.loads(response.read().decode("utf-8"))

        answer = (
            result.get("message", {}).get("content", "")
        ).strip()

        elapsed = time.perf_counter() - start_time

        print("\n" + "=" * 60)
        print("MEDINTEL AI - DOCUMENT ANSWER")
        print("=" * 60)

        print("\n" + answer)

        print("\n" + "-" * 60)

        print(f"Response time: {elapsed:.2f} seconds")

        print(
            "\nReminder: Generated explanations must be "
            "checked against their cited source passages."
        )

    except urllib.error.URLError as error:

        print("\nCould not connect to Ollama.")
        print("Make sure 'ollama serve' is running.")
        print("Error:", error)


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n" + "=" * 60)
    print("MEDINTEL AI - DOCUMENT QUESTION ANSWERING")
    print("=" * 60)

    embeddings, documents = load_rag_data()

    print("\nLoaded document chunks:", len(documents))
    print("Embedding dimensions:", embeddings.shape[1])
    print("LLM:", LLM_NAME)

    model = load_embedding_model()

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print("Question cannot be empty.")
        return

    print("\nSearching relevant document passages...")

    retrieved_chunks = retrieve_chunks(
        question,
        model,
        embeddings,
        documents
    )

    if not retrieved_chunks:

        print("No relevant passages found.")
        return

    display_sources(retrieved_chunks)
    display_source_warnings(retrieved_chunks)

    # Referral questions: direct extraction.
    if re.search(
            r"\b(referr\w*|refer\w*|higher healthcare|tertiary care)\b",
            question,
            re.IGNORECASE
        ):

            evidence = extract_referral_criteria(retrieved_chunks)

            if evidence:
                display_referral_answer(evidence)
                return

            print("\nCould not reliably extract referral criteria.")
            print("Please verify the original PDF page manually.")
            print("LLM fallback disabled for this question.")
            return

    # Patient review questions: direct extraction.
    if re.search(
        r"\b(review|reviewing|follow[- ]?up|monitoring)\b",
        question,
        re.IGNORECASE
    ):

        evidence = extract_review_evidence(retrieved_chunks)

        display_review_answer(evidence)
        return

    # Other questions: LLM answer.
    generate_llm_answer(
        question,
        retrieved_chunks
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:

        print("\nAn error occurred:")
        print(error)
