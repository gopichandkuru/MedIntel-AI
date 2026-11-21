
import json
import re
from pathlib import Path
from collections import Counter

base = Path("data/processed/medquad_v3")

patterns = {
    "possible_ocr_error": r"l000",
    "long_word": r"[a-z]{15,}",
    "repeated_dots": r"\.{5,}"
}

for split in ["train", "valid", "test"]:
    path = base / f"{split}.jsonl"

    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    examples = {name: [] for name in patterns}
    counts = Counter()

    for r in records:
        answer = r["answer"].casefold()

        for name, pattern in patterns.items():
            if re.search(pattern, answer):
                counts[name] += 1
                if len(examples[name]) < 5:
                    examples[name].append(r)

    print(f"\n========== {split.upper()} ==========")

    for name in patterns:
        print(f"\n{name}: {counts[name]} records")

        for r in examples[name]:
            print("Source:", r["source"])
            print("Question:", r["question"])
            print("Answer:", r["answer"][:500])
            print("-" * 50)
