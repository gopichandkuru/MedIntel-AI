
import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 1024

BASE = Path("data/processed/medquad")
OUTPUT = BASE / "sft"
OUTPUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

SYSTEM_PROMPT = (
    "You are a medical information assistant. "
    "Provide educational information based on the supplied medical reference. "
    "Do not invent facts, diagnose patients, or prescribe personalized treatment. "
    "If information is insufficient, clearly say so."
)

def prepare_split(split):
    source = BASE / f"{split}.jsonl"
    destination = OUTPUT / f"{split}_sft.jsonl"

    accepted = []
    rejected = []

    with source.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record["question"]},
                {"role": "assistant", "content": record["answer"]}
            ]

            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False
            )

            formatted = {
                "text": tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            }

            if len(token_ids) <= MAX_LENGTH:
                accepted.append(formatted)
            else:
                rejected.append({
                    "id": record["id"],
                    "token_length": len(token_ids),
                    "question": record["question"],
                    "answer": record["answer"]
                })

    with destination.open("w", encoding="utf-8") as f:
        for item in accepted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(accepted), rejected


train_count, train_long = prepare_split("train")
valid_count, valid_long = prepare_split("valid")

with (OUTPUT / "long_examples.jsonl").open(
    "w", encoding="utf-8"
) as f:
    for item in train_long + valid_long:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\nSFT DATA PREPARATION COMPLETE")
print("Training examples:", train_count)
print("Validation examples:", valid_count)
print("Long examples preserved:", len(train_long) + len(valid_long))
print("Output directory:", OUTPUT)
