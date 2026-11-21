
import json
from pathlib import Path
from collections import Counter

base = Path("data/processed/medquad_v3")
sources = Counter()
total = 0

for split in ["train", "valid", "test"]:
    path = base / f"{split}.jsonl"

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            answer = record["answer"].strip().casefold()

            resource = any(x in answer for x in [
                "these resources address",
                "these resources from medlineplus",
                "diagnostic tests - drug therapy",
                "gene review:"
            ])

            if not resource and 60 <= len(answer) <= 5000:
                sources[record["source"]] += 1
                total += 1

print("Total direct-answer candidates:", total)
print("\nSource distribution:")

for source, count in sources.most_common():
    percentage = count / total * 100
    print(f"{source}: {count} ({percentage:.2f}%)")
