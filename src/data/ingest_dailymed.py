#!/usr/bin/env python3
"""
Build a separate DailyMed medication-label retrieval store for MedIntel-AI.
Initial scope: metformin and glimepiride. Existing WHO and MedQuAD files are not modified.
"""
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed" / "dailymed" / "retrieval"
API_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
SPL_BASE = API_BASE
DRUGS = ["metformin", "glimepiride"]
LABELS_PER_DRUG = 3
PAGE_SIZE = 100
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
CHUNK_CHARS = 1400
CHUNK_OVERLAP = 180
NS = {"hl7": "urn:hl7-org:v3"}
RELEVANT_SECTION_TERMS = (
    "warning", "indications and usage", "dosage", "contraindications",
    "warnings and precautions", "adverse reactions", "drug interactions",
    "use in specific populations", "overdosage", "patient counseling",
    "clinical pharmacology", "description"
)

def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def get_label_candidates(session, drug):
    response = session.get(
        f"{API_BASE}.json",
        params={"drug_name": drug, "name_type": "generic", "pagesize": PAGE_SIZE, "page": 1},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    selected = []
    for item in payload.get("data", []):
        title = clean_text(item.get("title", "")).upper()
        if drug.upper() not in title or " AND " in title or not item.get("setid"):
            continue
        selected.append(item)
        if len(selected) >= LABELS_PER_DRUG:
            break
    return selected, payload.get("metadata", {}).get("total_elements", 0)

def extract_sections(xml_bytes):
    root = ET.fromstring(xml_bytes)
    sections = []
    for section in root.findall(".//hl7:section", NS):
        title_node = section.find("./hl7:title", NS)
        text_node = section.find("./hl7:text", NS)
        title = clean_text(" ".join(title_node.itertext())) if title_node is not None else ""
        content = clean_text(" ".join(text_node.itertext())) if text_node is not None else ""
        if title and content and any(term in title.lower() for term in RELEVANT_SECTION_TERMS):
            sections.append((title, content))
    return sections

def split_text(text, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "MedIntel-AI/1.0 (research ingestion)"})
    metadata_rows, record_rows, seen_content = [], [], set()
    downloaded_labels = 0

    for drug in DRUGS:
        print(f"\n[DailyMed] Searching: {drug}")
        labels, total = get_label_candidates(session, drug)
        print(f"  API reports {total} labels; selecting {len(labels)} recent single-ingredient labels")
        for label in labels:
            setid = label["setid"]
            title = clean_text(label.get("title", ""))
            published = label.get("published_date")
            source_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
            response = session.get(f"{SPL_BASE}/{setid}.xml", timeout=60)
            response.raise_for_status()
            sections = extract_sections(response.content)
            if not sections:
                print(f"  WARNING: no usable sections for {setid}")
                continue
            downloaded_labels += 1
            added = 0
            for section_title, section_text in sections:
                for chunk_index, chunk in enumerate(split_text(section_text)):
                    digest = hashlib.sha256(
                        f"{drug}|{section_title}|{chunk}".encode("utf-8")
                    ).hexdigest()
                    if digest in seen_content:
                        continue
                    seen_content.add(digest)
                    chunk_id = f"dailymed-{setid}-{len(metadata_rows) + 1}"
                    metadata_rows.append({
                        "chunk_id": chunk_id, "source": "DailyMed",
                        "source_type": "US medication prescribing label",
                        "drug_name": drug, "label_title": title, "setid": setid,
                        "published_date": published, "section": section_title,
                        "chunk_index": chunk_index, "source_url": source_url,
                        "country_scope": "United States",
                    })
                    record_rows.append({"chunk_id": chunk_id, "text": chunk,
                                        "content": chunk, "answer": chunk})
                    added += 1
            print(f"  {title[:90]} — {len(sections)} sections, {added} unique chunks")

    if not record_rows:
        raise RuntimeError("No DailyMed chunks extracted; existing stores were not changed.")

    print(f"\n[DailyMed] Embedding {len(record_rows)} chunks on CPU...")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    embeddings = model.encode(
        [row["text"] for row in record_rows], batch_size=BATCH_SIZE,
        show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    np.save(OUTPUT_DIR / "dailymed_embeddings.npy", embeddings)
    write_jsonl(OUTPUT_DIR / "dailymed_metadata.jsonl", metadata_rows)
    write_jsonl(OUTPUT_DIR / "dailymed_records.jsonl", record_rows)
    summary = {
        "source": "DailyMed", "drugs": DRUGS, "labels_downloaded": downloaded_labels,
        "chunks": len(record_rows), "embedding_shape": list(embeddings.shape),
        "embedding_model": EMBED_MODEL,
        "files": ["dailymed_embeddings.npy", "dailymed_metadata.jsonl", "dailymed_records.jsonl"],
    }
    (OUTPUT_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\\nSUCCESS: DailyMed store created:", OUTPUT_DIR)
    print("Labels:", downloaded_labels, "| Chunks:", len(record_rows), "| Embeddings:", embeddings.shape)
    print("WHO and MedQuAD data were not modified.")
    print("This script creates the DailyMed store; the live RAG loader must be connected separately.")

if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"\nNETWORK/API ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nINGESTION ERROR: {exc}", file=sys.stderr)
        raise
