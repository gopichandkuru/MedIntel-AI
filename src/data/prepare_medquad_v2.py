import json
import random
import re
import html
import hashlib
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RAW_DIR = Path("data/raw/medical_qa/MedQuAD")
OUTPUT_DIR = Path("data/processed/medquad_v2")

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 30
MAX_ANSWER_LENGTH = 6000


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
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def make_record_id(source, relative_path, question_id, qa_index):
    identity = (
        f"{source}|{relative_path}|{question_id}|{qa_index}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


# --------------------------------------------------
# XML PARSING
# --------------------------------------------------

def parse_xml_file(file_path):
    records = []

    try:
        root = ET.parse(file_path).getroot()

        document_id = root.attrib.get("id", "").strip()
        source = root.attrib.get("source", "").strip()
        source_url = root.attrib.get("url", "").strip()

        # Use folder name if XML source is missing.
        if not source:
            source = file_path.parent.name

        relative_path = file_path.relative_to(RAW_DIR).as_posix()

        focus_element = root.find("Focus")
        focus = clean_text(
            focus_element.text if focus_element is not None else ""
        )

        # A missing document ID must not group unrelated XML files.
        if document_id:
            source_document_id = f"{source}:{document_id}"
        else:
            source_document_id = f"{source}:{relative_path}"

        for qa_index, qa in enumerate(root.findall(".//QAPair"), start=1):
            question_element = qa.find("Question")
            answer_element = qa.find("Answer")

            if question_element is None or answer_element is None:
                continue

            question = clean_text(
                " ".join(question_element.itertext())
            )
            answer = clean_text(
                " ".join(answer_element.itertext())
            )

            question_type = question_element.attrib.get("qtype", "")
            question_id = question_element.attrib.get("qid", "")

            if len(question) < MIN_QUESTION_LENGTH:
                continue

            if len(answer) < MIN_ANSWER_LENGTH:
                continue

            if len(answer) > MAX_ANSWER_LENGTH:
                continue

            record_id = make_record_id(
                source,
                relative_path,
                question_id,
                qa_index
            )

            records.append({
                "id": question_id,
                "record_id": record_id,
                "document_id": document_id,
                "source_document_id": source_document_id,
                "xml_filename": relative_path,
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
# DUPLICATE AUDIT - NO RECORDS ARE DELETED
# --------------------------------------------------

def audit_duplicates(records):
    groups = defaultdict(list)

    for record in records:
        normalized = normalize_for_duplicate(record["question"])

        if normalized:
            groups[normalized].append(record)

    duplicate_groups = []

    for normalized, group_records in groups.items():
        if len(group_records) > 1:
            duplicate_groups.append({
                "normalized_question": normalized,
                "record_count": len(group_records),
                "records": [
                    {
                        "record_id": record["record_id"],
                        "id": record["id"],
                        "source": record["source"],
                        "document_id": record["document_id"],
                        "xml_filename": record["xml_filename"],
                        "question": record["question"],
                        "source_url": record["source_url"]
                    }
                    for record in group_records
                ]
            })

    return duplicate_groups


# --------------------------------------------------
# GROUP-AWARE DATA SPLIT
# --------------------------------------------------

def split_dataset(records):
    groups = defaultdict(list)

    for record in records:
        groups[record["source_document_id"]].append(record)

    group_ids = sorted(groups.keys())

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(group_ids)

    total_groups = len(group_ids)

    train_end = int(total_groups * TRAIN_RATIO)
    valid_end = train_end + int(total_groups * VALID_RATIO)

    train_ids = set(group_ids[:train_end])
    valid_ids = set(group_ids[train_end:valid_end])
    test_ids = set(group_ids[valid_end:])

    train = [
        record for key in train_ids for record in groups[key]
    ]
    valid = [
        record for key in valid_ids for record in groups[key]
    ]
    test = [
        record for key in test_ids for record in groups[key]
    ]

    return train, valid, test


# --------------------------------------------------
# SAVE JSONL AND JSON
# --------------------------------------------------

def save_jsonl(records, file_path):
    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def save_json(data, file_path):
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_splits(train, valid, test):
    train_groups = {r["source_document_id"] for r in train}
    valid_groups = {r["source_document_id"] for r in valid}
    test_groups = {r["source_document_id"] for r in test}

    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)

    all_records = train + valid + test

    record_ids = [r["record_id"] for r in all_records]

    assert len(record_ids) == len(set(record_ids)), (
        "Duplicate record_id values detected"
    )

    assert len(all_records) == len(train) + len(valid) + len(test)

    return {
        "record_ids_unique": True,
        "source_document_group_overlap": 0,
        "all_records_preserved_in_splits": True
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DIR}")

    # Safety: never overwrite an existing v2 output directory.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Choose a new output directory before running again."
        )

    print("\nLoading MedQuAD dataset...")
    records = load_dataset()

    print(f"\nAccepted records: {len(records)}")

    if not records:
        raise RuntimeError("No valid records were loaded.")

    print("\nAuditing duplicate questions...")
    duplicate_groups = audit_duplicates(records)

    print(f"Duplicate question groups: {len(duplicate_groups)}")

    print("\nSplitting by source-aware parent document...")
    train, valid, test = split_dataset(records)

    validation = validate_splits(train, valid, test)

    # Create output only after data processing and validation succeed.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid, OUTPUT_DIR / "valid.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    save_jsonl(
        duplicate_groups,
        OUTPUT_DIR / "duplicate_question_groups.jsonl"
    )

    source_counts = defaultdict(int)
    for record in records:
        source_counts[record["source"]] += 1

    summary = {
        "random_seed": RANDOM_SEED,
        "total_records": len(records),
        "train_records": len(train),
        "validation_records": len(valid),
        "test_records": len(test),
        "unique_source_document_groups": len({
            r["source_document_id"] for r in records
        }),
        "duplicate_question_groups": len(duplicate_groups),
        "duplicate_records_in_groups": sum(
            group["record_count"] for group in duplicate_groups
        ),
        "source_record_counts": dict(sorted(source_counts.items())),
        "validation": validation,
        "deduplication_applied": False,
        "output_directory": str(OUTPUT_DIR)
    }

    save_json(summary, OUTPUT_DIR / "audit_summary.json")

    print("\n========== V2 DATASET SUMMARY ==========")
    print(f"Training records:       {len(train)}")
    print(f"Validation records:     {len(valid)}")
    print(f"Test records:            {len(test)}")
    print(f"Total records:           {len(records)}")
    print(f"Duplicate groups:        {len(duplicate_groups)}")
    print(f"Unique source-doc groups: {summary['unique_source_document_groups']}")
    print(f"Unique record IDs:       {validation['record_ids_unique']}")
    print(f"Group leakage:           {validation['source_document_group_overlap']}")
    print(f"Output directory:        {OUTPUT_DIR}")

    print("\nPreprocessing V2 completed successfully.")


if __name__ == "__main__":
    main()
