
import json
from pathlib import Path
from collections import Counter

base = Path("data/processed/medquad_v3")
splits = ["train", "valid", "test"]

required_fields = [
    "record_id",
    "id",
    "document_id",
    "question",
    "answer",
    "question_type",
    "source",
    "source_url",
    "source_document_id"
]

all_ids = {}
total_records = 0

for split in splits:
    path = base / f"{split}.jsonl"

    if not path.exists():
        print(f"MISSING FILE: {path}")
        continue

    records = []
    invalid_json = 0
    missing_fields = Counter()
    missing_urls = 0
    duplicate_ids = 0

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            records.append(record)

            for field in required_fields:
                if not record.get(field):
                    missing_fields[field] += 1

            if not record.get("source_url"):
                missing_urls += 1

            record_id = record.get("record_id")

            if record_id:
                if record_id in all_ids:
                    duplicate_ids += 1
                    print(
                        "Duplicate record ID:",
                        record_id,
                        "in", split,
                        "previously in", all_ids[record_id]
                    )
                else:
                    all_ids[record_id] = split

    total_records += len(records)

    print(f"\n========== {split.upper()} ==========")
    print("Records:", len(records))
    print("Invalid JSON lines:", invalid_json)
    print("Missing required fields:", dict(missing_fields))
    print("Missing source URLs:", missing_urls)
    print("Duplicate record IDs:", duplicate_ids)

print("\n========== FINAL SUMMARY ==========")
print("Total valid JSON records:", total_records)
print("Unique record IDs:", len(all_ids))
print("Expected records: 16152")
print("Record count correct:", total_records == 16152)
