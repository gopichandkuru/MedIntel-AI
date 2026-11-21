import json
import hashlib
import random
import re
from pathlib import Path
from collections import defaultdict

INPUT_DIR = Path("data/processed/medquad_v2")
OUTPUT_DIR = Path("data/processed/medquad_v3")

SEED = 42
RATIOS = {
    "train": 0.80,
    "valid": 0.10,
    "test": 0.10
}


def normalize_question(text):
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def load_records():
    records = []

    for split in ["train", "valid", "test"]:
        path = INPUT_DIR / f"{split}.jsonl"

        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    return records


def create_components(records):
    uf = UnionFind(len(records))

    document_owner = {}
    question_owner = {}

    for index, record in enumerate(records):
        document_key = record["source_document_id"]
        question_key = normalize_question(record["question"])

        if document_key in document_owner:
            uf.union(index, document_owner[document_key])
        else:
            document_owner[document_key] = index

        if question_key:
            if question_key in question_owner:
                uf.union(index, question_owner[question_key])
            else:
                question_owner[question_key] = index

    components = defaultdict(list)

    for index, record in enumerate(records):
        components[uf.find(index)].append(record)

    return list(components.values())


def assign_splits(components):
    total = sum(len(group) for group in components)

    targets = {
        name: total * ratio
        for name, ratio in RATIOS.items()
    }

    counts = {name: 0 for name in RATIOS}
    assigned = {name: [] for name in RATIOS}

    # Largest connected groups are assigned first.
    components = sorted(
        components,
        key=lambda group: (
            -len(group),
            min(r["record_id"] for r in group)
        )
    )

    for group in components:
        size = len(group)

        # Select the split with the lowest normalized
        # incremental squared deviation from its target.
        costs = {}

        for name in RATIOS:
            old = counts[name]
            target = targets[name]

            costs[name] = (
                ((old + size - target) ** 2 - (old - target) ** 2)
                / max(target, 1)
            )

        selected = min(costs, key=lambda name: (costs[name], name))

        assigned[selected].extend(group)
        counts[selected] += size

    return assigned, counts


def validate_splits(splits):
    question_splits = defaultdict(set)
    document_splits = defaultdict(set)
    record_ids = []

    for split_name, records in splits.items():
        for record in records:
            question_key = normalize_question(record["question"])
            question_splits[question_key].add(split_name)
            document_splits[record["source_document_id"]].add(split_name)
            record_ids.append(record["record_id"])

    question_leakage = sum(
        1 for names in question_splits.values() if len(names) > 1
    )

    document_leakage = sum(
        1 for names in document_splits.values() if len(names) > 1
    )

    duplicate_ids = len(record_ids) - len(set(record_ids))

    assert question_leakage == 0
    assert document_leakage == 0
    assert duplicate_ids == 0

    return {
        "cross_split_question_groups": question_leakage,
        "cross_split_document_groups": document_leakage,
        "duplicate_record_ids": duplicate_ids
    }


def save_jsonl(records, path):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Choose another output directory before running again."
        )

    print("Loading V2 records...")
    records = load_records()
    print("Records loaded:", len(records))

    if not records:
        raise RuntimeError("No records found.")

    print("\nBuilding connected components...")
    components = create_components(records)

    sizes = sorted(
        (len(group) for group in components),
        reverse=True
    )

    print("Connected components:", len(components))
    print("Largest component:", sizes[0])
    print("Top 10 component sizes:", sizes[:10])

    print("\nAssigning components to splits...")
    splits, counts = assign_splits(components)

    validation = validate_splits(splits)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    for name in ["train", "valid", "test"]:
        save_jsonl(
            splits[name],
            OUTPUT_DIR / f"{name}.jsonl"
        )

    component_report = []

    for group in components:
        group_id = hashlib.sha256(
            "|".join(sorted(r["record_id"] for r in group)).encode()
        ).hexdigest()

        component_report.append({
            "split_group_id": group_id,
            "record_count": len(group),
            "source_documents": sorted({
                r["source_document_id"] for r in group
            }),
            "normalized_questions": sorted({
                normalize_question(r["question"]) for r in group
            })
        })

    with (OUTPUT_DIR / "component_audit.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for item in component_report:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "input_records": len(records),
        "output_records": sum(counts.values()),
        "component_count": len(components),
        "largest_component": sizes[0],
        "component_sizes_top_10": sizes[:10],
        "split_counts": counts,
        "split_percentages": {
            name: round(counts[name] / len(records) * 100, 2)
            for name in counts
        },
        "validation": validation,
        "seed": SEED
    }

    with (OUTPUT_DIR / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    assert sum(counts.values()) == len(records)

    print("\n========== V3 RESULTS ==========")
    print("Total records:", len(records))

    for name in ["train", "valid", "test"]:
        print(
            f"{name}: {counts[name]} "
            f"({summary['split_percentages'][name]}%)"
        )

    print("\nLargest component:", sizes[0])
    print("Validation:", validation)
    print("Output:", OUTPUT_DIR)
    print("\nV3 splitting completed successfully.")


if __name__ == "__main__":
    main()
