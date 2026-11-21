#!/usr/bin/env python3
"""
Build a separate, provenance-preserving DailyMed retrieval store.
Existing WHO and MedQuAD stores are not modified.
"""

import hashlib
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data/processed/dailymed/retrieval"

API_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
DRUGS = ["metformin", "glimepiride"]

LABELS_PER_DRUG = 3
PAGE_SIZE = 100
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16

CHUNK_CHARS = 1400
CHUNK_OVERLAP = 180

NS = {"hl7": "urn:hl7-org:v3"}

RELEVANT_SECTION_TERMS = (
    "warning",
    "hypoglycemia",
    "hypersensitivity",
    "hemolytic anemia",
    "cardiovascular mortality",
    "macrovascular outcomes",
    "indications and usage",
    "dosage",
    "contraindications",
    "adverse reactions",
    "clinical studies",
    "drug interactions",
    "use in specific populations",
    "overdosage",
    "patient counseling",
    "clinical pharmacology",
    "description",
)


def clean_text(value):
    return re.sub(r"[ \t]+", " ", value or "").strip()


def normalize_date(value):
    if not value:
        return datetime.min

    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue

    return datetime.min


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_all_candidates(session, drug):
    """Fetch all API pages so label selection is not limited to page one."""
    results = []
    page = 1
    total = None

    while total is None or len(results) < total:
        response = session.get(
            f"{API_BASE}.json",
            params={
                "drug_name": drug,
                "name_type": "generic",
                "pagesize": PAGE_SIZE,
                "page": page,
            },
            timeout=45,
        )
        response.raise_for_status()

        payload = response.json()
        results.extend(payload.get("data", []))

        metadata = payload.get("metadata", {})
        total = metadata.get("total_elements", len(results))

        if not payload.get("data"):
            break

        page += 1

    # Keep single-ingredient labels and preserve their individual identities.
    candidates = {}
    for item in results:
        title = clean_text(item.get("title", ""))
        setid = item.get("setid")

        if not setid or drug.lower() not in title.lower():
            continue

        if " AND " in title.upper():
            continue

        candidates[setid] = item

    selected = sorted(
        candidates.values(),
        key=lambda item: normalize_date(item.get("published_date")),
        reverse=True,
    )

    return selected[:LABELS_PER_DRUG], total


def render_xml_text(node):
    """
    Preserve table rows and columns instead of flattening all XML text.
    This is important for medication adverse-effect percentages.
    """
    output = []

    def walk(element):
        tag = element.tag.split("}")[-1].lower()

        if tag == "table":
            rows = []
            for tr in element.iter():
                if tr.tag.split("}")[-1].lower() != "tr":
                    continue

                cells = []
                for cell in list(tr):
                    cell_tag = cell.tag.split("}")[-1].lower()
                    if cell_tag in ("td", "th"):
                        value = clean_text(" ".join(cell.itertext()))
                        cells.append(value)

                if cells:
                    rows.append(" | ".join(cells))

            if rows:
                output.append("\n[TABLE]\n")
                output.extend(row + "\n" for row in rows)
                output.append("[END TABLE]\n")
            return

        if element.text:
            output.append(element.text)

        for child in list(element):
            walk(child)
            if child.tail:
                output.append(child.tail)

    walk(node)

    # Keep line boundaries; do not flatten tables into one sentence.
    lines = []
    for line in "".join(output).splitlines():
        line = clean_text(line)
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def extract_sections(xml_bytes):
    root = ET.fromstring(xml_bytes)
    sections = []

    for section in root.findall(".//hl7:section", NS):
        title_node = section.find("./hl7:title", NS)
        text_node = section.find("./hl7:text", NS)

        title = (
            clean_text(" ".join(title_node.itertext()))
            if title_node is not None else ""
        )

        content = render_xml_text(text_node) if text_node is not None else ""

        if (
            title and content
            and any(term in title.lower() for term in RELEVANT_SECTION_TERMS)
        ):
            sections.append((title, content))

    return sections


def split_text(text):
    """Split on line boundaries first to avoid breaking table rows."""
    if len(text) <= CHUNK_CHARS:
        return [text] if text.strip() else []

    chunks = []
    current = []

    for line in text.splitlines():
        candidate = "\n".join(current + [line])

        if len(candidate) > CHUNK_CHARS and current:
            piece = "\n".join(current).strip()
            if piece:
                chunks.append(piece)

            # Retain overlap only at complete line boundaries.
            overlap_lines = []
            overlap_size = 0
            for previous in reversed(current):
                if overlap_size + len(previous) > CHUNK_OVERLAP:
                    break
                overlap_lines.insert(0, previous)
                overlap_size += len(previous) + 1

            current = overlap_lines

        if len(line) > CHUNK_CHARS:
            if current:
                chunks.append("\n".join(current).strip())
                current = []

            start = 0
            while start < len(line):
                chunks.append(line[start:start + CHUNK_CHARS])
                start += CHUNK_CHARS - CHUNK_OVERLAP
        else:
            current.append(line)

    if current:
        piece = "\n".join(current).strip()
        if piece:
            chunks.append(piece)

    return [chunk for chunk in chunks if chunk]



def write_embeddings(path, embeddings):
    with path.open("wb") as file:
        np.save(file, embeddings)


def atomic_write_store(metadata_rows, record_rows, embeddings):
    """Write temporary files first; replace the store only after success."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "dailymed_embeddings.npy": lambda p: write_embeddings(p, embeddings),
        "dailymed_metadata.jsonl": lambda p: write_jsonl(p, metadata_rows),
        "dailymed_records.jsonl": lambda p: write_jsonl(p, record_rows),
    }

    temporary = {}

    try:
        for filename, writer in files.items():
            fd, temp_path = tempfile.mkstemp(
                prefix=filename + ".",
                suffix=".tmp",
                dir=OUTPUT_DIR,
            )
            os.close(fd)
            temp_path = Path(temp_path)
            temporary[filename] = temp_path
            writer(temp_path)

        for filename, temp_path in temporary.items():
            os.replace(temp_path, OUTPUT_DIR / filename)

    finally:
        for temp_path in temporary.values():
            if temp_path.exists():
                temp_path.unlink()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "MedIntel-AI/1.0 (medical research ingestion)"
    })

    metadata_rows = []
    record_rows = []
    seen_chunks = set()
    downloaded_labels = 0

    for drug in DRUGS:
        print(f"\n[DailyMed] Searching: {drug}")

        labels, total = fetch_all_candidates(session, drug)

        print(
            f"  API reports {total} labels; "
            f"selected {len(labels)} latest single-ingredient labels"
        )

        for label in labels:
            setid = label["setid"]
            title = clean_text(label.get("title", ""))
            published = label.get("published_date")

            source_url = (
                "https://dailymed.nlm.nih.gov/dailymed/"
                f"drugInfo.cfm?setid={setid}"
            )

            response = session.get(
                f"{API_BASE}/{setid}.xml",
                timeout=60,
            )
            response.raise_for_status()

            sections = extract_sections(response.content)

            if not sections:
                print(f"  WARNING: no usable sections for {setid}")
                continue

            downloaded_labels += 1
            added = 0

            print(f"\n  LABEL: {title}")
            print(f"  Published: {published}")

            for section_title, section_text in sections:
                chunks = split_text(section_text)

                for chunk_index, chunk in enumerate(chunks):
                    # Include setid so different labels never lose provenance.
                    digest = hashlib.sha256(
                        (
                            f"{drug}|{setid}|{section_title}|{chunk}"
                        ).encode("utf-8")
                    ).hexdigest()

                    if digest in seen_chunks:
                        continue

                    seen_chunks.add(digest)

                    chunk_id = (
                        f"dailymed-{setid}-{len(metadata_rows) + 1}"
                    )

                    metadata_rows.append({
                        "chunk_id": chunk_id,
                        "source": "DailyMed",
                        "source_type": "US medication prescribing label",
                        "drug_name": drug,
                        "label_title": title,
                        "setid": setid,
                        "published_date": published,
                        "section": section_title,
                        "chunk_index": chunk_index,
                        "source_url": source_url,
                        "country_scope": "United States",
                    })

                    record_rows.append({
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "content": chunk,
                        "answer": chunk,
                    })

                    added += 1

                print(
                    f"    {section_title}: "
                    f"{len(section_text)} characters, "
                    f"{len(chunks)} chunks"
                )

            print(f"  Added {added} unique chunks")

    if not record_rows:
        raise RuntimeError(
            "No DailyMed chunks extracted. Existing store was not replaced."
        )

    print(
        f"\n[DailyMed] Embedding {len(record_rows)} chunks on CPU..."
    )

    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    embeddings = model.encode(
        [row["text"] for row in record_rows],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if (
        embeddings.ndim != 2
        or len(embeddings) != len(record_rows)
        or not np.isfinite(embeddings).all()
    ):
        raise RuntimeError("Embedding validation failed. Store not replaced.")

    atomic_write_store(metadata_rows, record_rows, embeddings)

    summary = {
        "source": "DailyMed",
        "drugs": DRUGS,
        "labels_downloaded": downloaded_labels,
        "chunks": len(record_rows),
        "embedding_shape": list(embeddings.shape),
        "embedding_model": EMBED_MODEL,
        "files": [
            "dailymed_embeddings.npy",
            "dailymed_metadata.jsonl",
            "dailymed_records.jsonl",
        ],
    }

    summary_path = OUTPUT_DIR / "build_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSUCCESS: DailyMed store created:", OUTPUT_DIR)
    print("Labels:", downloaded_labels)
    print("Chunks:", len(record_rows))
    print("Embeddings:", embeddings.shape)
    print("WHO and MedQuAD data were not modified.")
    print("Existing RAG loader filenames and metadata fields are preserved.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"\nNETWORK/API ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nINGESTION ERROR: {exc}", file=sys.stderr)
        raise
