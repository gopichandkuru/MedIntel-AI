
import json
import re
from pathlib import Path
from collections import Counter

base = Path("data/processed/medquad_v3")

for split in ["train", "valid", "test"]:
    path = base / f"{split}.jsonl"

    total = 0
    repeated_count = 0
    examples = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            total += 1
            answer = record["answer"]

            sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
            normalized = [
                re.sub(r'\W+', ' ', s.casefold()).strip()
                for s in sentences
            ]

            counts = Counter(s for s in normalized if len(s) > 30)
            repeated = [s for s, count in counts.items() if count > 1]

            if repeated:
                repeated_count += 1

                if len(examples) < 5:
                    examples.append((record, repeated))

    print(f"\n========== {split.upper()} ==========")
    print("Total records:", total)
    print("Answers with repeated sentences:", repeated_count)

    for record, repeated in examples:
        print("\nSource:", record["source"])
        print("Question:", record["question"])
        print("Repeated sentences:")
        for sentence in repeated:
            print("-", sentence[:300])
        print("-" * 60)
