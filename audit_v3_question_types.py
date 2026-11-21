
import json
from pathlib import Path
from collections import Counter

base = Path("data/processed/medquad_v3")

for split in ["train", "valid", "test"]:
    counts = Counter()

    with (base / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            counts[record.get("question_type", "unknown")] += 1

    print(f"\n========== {split.upper()} ==========")
    for question_type, count in counts.most_common():
        print(f"{question_type}: {count}")
