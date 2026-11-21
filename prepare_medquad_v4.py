
import json
import re
from pathlib import Path
from collections import Counter

SOURCE_DIR = Path("data/processed/medquad_v3")
OUTPUT_DIR = Path("data/processed/medquad_v4")

SPLITS = ["train", "valid", "test"]

RESOURCE_PATTERNS = [
    "these resources address",
    "these resources from medlineplus",
    "diagnostic tests - drug therapy",
    "gene review:",
]

def classify_record(record):
    answer = record.get("answer", "").strip()
    normalized = re.sub(r"\s+", " ", answer).lower()

    if any(pattern in normalized for pattern in RESOURCE_PATTERNS):
        return "resource_lists"

    if len(answer) < 60 or len(answer) > 5000:
        return "review_required"

    return "direct_answers"


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. Nothing was overwritten."
        )

    categories = [
        "direct_answers",
        "resource_lists",
        "review_required",
    ]

    records_by_category = {
        category: {split: [] for split in SPLITS}
        for category in categories
    }

    summary = {
        "source": str(SOURCE_DIR),
        "output": str(OUTPUT_DIR),
        "classification_rules": {
            "resource_lists": RESOURCE_PATTERNS,
            "review_required": "Answer length below 60 or above 5000 characters",
            "direct_answers": "All remaining records; not clinically verified",
        },
        "splits": {},
        "totals": {},
    }

    all_record_ids = set()
    total = 0

    for split in SPLITS:
        input_file = SOURCE_DIR / f"{split}.jsonl"

        if not input_file.exists():
            raise FileNotFoundError(f"Missing input file: {input_file}")

        split_counts = Counter()

        with input_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {input_file}, line {line_number}: {error}"
                    )

                record_id = record.get("record_id")

                if not record_id:
                    raise ValueError(
                        f"Missing record_id in {input_file}, line {line_number}"
                    )

                if record_id in all_record_ids:
                    raise ValueError(f"Duplicate record ID: {record_id}")

                all_record_ids.add(record_id)

                category = classify_record(record)

                # Preserve the original record without modifying its fields.
                records_by_category[category][split].append(record)

                split_counts[category] += 1
                total += 1

        summary["splits"][split] = {
            category: split_counts[category]
            for category in categories
        }
        summary["splits"][split]["total"] = sum(split_counts.values())

    expected_total = 16152

    if total != expected_total:
        raise ValueError(
            f"Expected {expected_total} records, but found {total}. "
            "Output was not created."
        )

    for category in categories:
        summary["totals"][category] = sum(
            len(records_by_category[category][split])
            for split in SPLITS
        )

    summary["total_records"] = total
    summary["unique_record_ids"] = len(all_record_ids)
    summary["records_modified"] = False
    summary["clinical_validation_performed"] = False

    # Create output only after all input records pass validation.
    OUTPUT_DIR.mkdir(parents=True)

    for category in categories:
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir()

        for split in SPLITS:
            output_file = category_dir / f"{split}.jsonl"

            with output_file.open("w", encoding="utf-8") as file:
                for record in records_by_category[category][split]:
                    file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

    audit_dir = OUTPUT_DIR / "audit"
    audit_dir.mkdir()

    with (audit_dir / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n========== MEDQUAD V4 SUMMARY ==========")

    for split in SPLITS:
        print(f"\n{split.upper()}")
        for category in categories:
            print(f"{category}: {summary['splits'][split][category]}")

    print("\n========== TOTALS ==========")
    for category in categories:
        print(f"{category}: {summary['totals'][category]}")

    print("\nTotal records:", total)
    print("Unique record IDs:", len(all_record_ids))
    print("Original V3 modified: No")
    print("Clinical validation performed: No")
    print("V4 output:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
