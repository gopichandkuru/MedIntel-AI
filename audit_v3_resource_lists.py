
import json
import re
from pathlib import Path
from collections import Counter

base = Path("data/processed/medquad_v3")

patterns = [
    r"these resources address",
    r"these resources from medlineplus",
    r"diagnostic tests\s*-\s*drug therapy",
    r"gene review:"
]

for split in ["train", "valid", "test"]:
    source_counts = Counter()
    type_counts = Counter()
    examples = []

    with (base / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            answer = record["answer"].casefold()

            if any(re.search(p, answer) for p in patterns):
                source_counts[record["source"]] += 1
                type_counts[record["question_type"]] += 1

                if len(examples) < 3:
                    examples.append(record)

    print(f"\n========== {split.upper()} ==========")
    print("Resource-list answers by source:")
    for source, count in source_counts.most_common():
        print(source, ":", count)

    print("\nBy question type:")
    for question_type, count in type_counts.most_common():
        print(question_type, ":", count)

    print("\nExamples:")
    for record in examples:
        print("Question:", record["question"])
        print("Answer:", record["answer"][:400])
        print("-" * 50)
