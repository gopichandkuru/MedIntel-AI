
import json
import random
import re
import html
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RAW_DIR = Path("data/raw/medical_qa/MedQuAD")
OUTPUT_DIR = Path("data/processed/medquad")

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 30
MAX_ANSWER_LENGTH = 6000

random.seed(RANDOM_SEED)


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_duplicate(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# XML PARSING
# --------------------------------------------------

def parse_xml_file(file_path):
    records = []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        document_id = root.attrib.get("id", "")
        source = root.attrib.get("source", "")
        source_url = root.attrib.get("url", "")

        focus_element = root.find("Focus")
        focus = clean_text(
            focus_element.text if focus_element is not None else ""
        )

        for qa in root.findall(".//QAPair"):
            question_element = qa.find("Question")
            answer_element = qa.find("Answer")

            if question_element is None or answer_element is None:
                continue

            question = clean_text(" ".join(question_element.itertext()))
            answer = clean_text(" ".join(answer_element.itertext()))

            question_type = question_element.attrib.get("qtype", "")
            question_id = question_element.attrib.get("qid", "")

            if len(question) < MIN_QUESTION_LENGTH:
                continue

            if len(answer) < MIN_ANSWER_LENGTH:
                continue

            if len(answer) > MAX_ANSWER_LENGTH:
                continue

            records.append({
                "id": question_id,
                "document_id": document_id,
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "focus": focus,
                "source": source,
                "source_url": source_url,
                "license": "CC BY 4.0 - verify source-specific terms"
            })

    except (ET.ParseError, OSError) as error:
        print(f"Skipping {file_path}: {error}")

    return records


# --------------------------------------------------
# LOAD ALL RECORDS
# --------------------------------------------------

def load_dataset():
    all_records = []

    xml_files = sorted(RAW_DIR.rglob("*.xml"))

    print(f"Found XML files: {len(xml_files)}")

    for index, file_path in enumerate(xml_files, start=1):
        records = parse_xml_file(file_path)
        all_records.extend(records)

        if index % 1000 == 0:
            print(
                f"Processed {index}/{len(xml_files)} files | "
                f"QA pairs: {len(all_records)}"
            )

    return all_records


# --------------------------------------------------
# DEDUPLICATION
# --------------------------------------------------

def remove_duplicates(records):
    unique_records = []
    seen = set()

    for record in records:
        key = normalize_for_duplicate(record["question"])

        if not key or key in seen:
            continue

        seen.add(key)
        unique_records.append(record)

    return unique_records


# --------------------------------------------------
# GROUP-AWARE DATA SPLIT
# --------------------------------------------------

def split_dataset(records):
    # Keep questions from the same XML document together.
    groups = {}

    for record in records:
        document_id = record["document_id"]

        if document_id not in groups:
            groups[document_id] = []

        groups[document_id].append(record)

    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    total_groups = len(group_ids)

    train_end = int(total_groups * TRAIN_RATIO)
    valid_end = train_end + int(total_groups * VALID_RATIO)

    train_ids = group_ids[:train_end]
    valid_ids = group_ids[train_end:valid_end]
    test_ids = group_ids[valid_end:]

    train = [r for key in train_ids for r in groups[key]]
    valid = [r for key in valid_ids for r in groups[key]]
    test = [r for key in test_ids for r in groups[key]]

    return train, valid, test


# --------------------------------------------------
# SAVE JSONL
# --------------------------------------------------

def save_jsonl(records, file_path):
    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading MedQuAD dataset...")

    records = load_dataset()

    print(f"\nRaw QA pairs: {len(records)}")

    records = remove_duplicates(records)

    print(f"After question deduplication: {len(records)}")

    random.seed(RANDOM_SEED)
    train, valid, test = split_dataset(records)

    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid, OUTPUT_DIR / "valid.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    print("\n========== DATASET SUMMARY ==========")
    print(f"Training records:   {len(train)}")
    print(f"Validation records: {len(valid)}")
    print(f"Test records:       {len(test)}")
    print(f"Total records:      {len(records)}")
    print(f"Output directory:   {OUTPUT_DIR}")

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
