
import json
import re
from pathlib import Path
from collections import Counter

base = Path("data/processed/medquad_v3")

resource_patterns = [
    r"these resources address",
    r"these resources from medlineplus",
    r"diagnostic tests\s*-\s*drug therapy",
    r"gene review:"
]

def classify(record):
    answer = record["answer"].strip()
    lower = answer.casefold()

    if any(re.search(p, lower) for p in resource_patterns):
        return "resource_list"

    if len(answer) < 60:
        return "short_answer_review"

    if len(answer) > 5000:
        return "long_answer_review"

    return "direct_answer_candidate"

for split in ["train", "valid", "test"]:
    counts = Counter()
    source_counts = Counter()

    path = base / f"{split}.jsonl"

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            category = classify(record)

            counts[category] += 1
            source_counts[(record["source"], category)] += 1

    print(f"\n========== {split.upper()} ==========")

    for category, count in counts.most_common():
        print(f"{category}: {count}")

    print("\nSource-wise classification:")
    for (source, category), count in sorted(source_counts.items()):
        print(f"{source} | {category}: {count}")
