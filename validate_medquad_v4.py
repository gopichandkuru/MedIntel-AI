
import json
from pathlib import Path
from collections import Counter

V3_DIR = Path("data/processed/medquad_v3")
V4_DIR = Path("data/processed/medquad_v4")

SPLITS = ["train", "valid", "test"]
CATEGORIES = ["direct_answers", "resource_lists", "review_required"]

RESOURCE_PATTERNS = [
    "these resources address",
    "these resources from medlineplus",
    "diagnostic tests - drug therapy",
    "gene review:",
]


def classify_record(record):
    answer = record.get("answer", "").strip()
    normalized = " ".join(answer.lower().split())

    if any(pattern in normalized for pattern in RESOURCE_PATTERNS):
        return "resource_lists"

    if len(answer) < 60 or len(answer) > 5000:
        return "review_required"

    return "direct_answers"


def load_jsonl(path):
    records = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON: {path}, line {line_number}: {error}"
                )

    return records


def main():
    errors = []
    total_v3 = 0
    total_v4 = 0

    for split in SPLITS:
        original_records = load_jsonl(V3_DIR / f"{split}.jsonl")
        original_by_id = {
            record["record_id"]: record for record in original_records
        }

        if len(original_by_id) != len(original_records):
            errors.append(f"Duplicate IDs in V3 {split}")

        v4_by_id = {}
        category_counts = Counter()

        for category in CATEGORIES:
            path = V4_DIR / category / f"{split}.jsonl"
            records = load_jsonl(path)
            category_counts[category] = len(records)

            for record in records:
                record_id = record.get("record_id")

                if not record_id:
                    errors.append(f"Missing record ID in {path}")
                    continue

                if record_id in v4_by_id:
                    errors.append(f"Duplicate V4 ID: {record_id}")
                    continue

                v4_by_id[record_id] = (record, category)

        missing = set(original_by_id) - set(v4_by_id)
        extra = set(v4_by_id) - set(original_by_id)

        if missing:
            errors.append(f"{split}: {len(missing)} missing records")

        if extra:
            errors.append(f"{split}: {len(extra)} unexpected records")

        for record_id in set(original_by_id) & set(v4_by_id):
            original = original_by_id[record_id]
            copied, category = v4_by_id[record_id]

            if original != copied:
                errors.append(f"Modified record: {record_id}")

            expected_category = classify_record(original)

            if category != expected_category:
                errors.append(
                    f"Wrong category for {record_id}: "
                    f"{category}, expected {expected_category}"
                )

        total_v3 += len(original_records)
        total_v4 += len(v4_by_id)

        print(f"\n========== {split.upper()} ==========")
        print("V3 records:", len(original_records))
        print("V4 records:", len(v4_by_id))
        print("Direct answers:", category_counts["direct_answers"])
        print("Resource lists:", category_counts["resource_lists"])
        print("Review required:", category_counts["review_required"])
        print("Missing records:", len(missing))
        print("Extra records:", len(extra))

    print("\n========== FINAL VALIDATION ==========")
    print("V3 total:", total_v3)
    print("V4 total:", total_v4)
    print("Expected total: 16152")
    print("Errors:", len(errors))

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors[:30]:
            print("-", error)
    else:
        print("\nVALIDATION PASSED")
        print("All records preserved.")
        print("All categories correct.")
        print("Original V3 records unchanged.")


if __name__ == "__main__":
    main()
