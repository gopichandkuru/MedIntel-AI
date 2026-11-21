
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "models/medquad_qlora_final"

SYSTEM_PROMPT = """
You are MedIntel-AI, a medical information assistant.

Rules:
1. Provide educational medical information only.
2. Never diagnose a user's disease or claim they have a condition.
3. Never prescribe personalized medicines or dosages.
4. Never invent medical facts, tests, or treatments.
5. If information is uncertain, clearly say so.
6. Encourage users with concerning symptoms to consult a qualified
   healthcare professional.
7. For emergencies, advise contacting local emergency services.
8. Give clear, concise, evidence-based answers.
"""

# Basic protection for direct personalized diagnosis or prescription requests.
UNSAFE_PATTERNS = [
    r"\bdiagnose me\b",
    r"\bcan you diagnose\b",
    r"\bwhat disease do i have\b",
    r"\bwhat illness do i have\b",
    r"\bdo i have (a|any|the) disease\b",
    r"\bprescribe (me|medicine for me)\b",
    r"\bwhat dosage should i take\b",
    r"\bhow much medicine should i take\b",
]

def requires_refusal(question):
    question = question.lower()
    return any(
        re.search(pattern, question)
        for pattern in UNSAFE_PATTERNS
    )


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Configuring 4-bit quantization...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

print("Loading trained LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

# Disable sampling settings for deterministic generation.
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None

print("\nMedIntel-AI Medical QA Test")
print("Type 'exit' to quit.\n")

while True:
    question = input("\nYour medical question: ").strip()

    if question.lower() in ("exit", "quit"):
        print("Closing MedIntel-AI...")
        break

    if not question:
        continue

    # Refuse direct personalized diagnosis and prescription requests.
    if requires_refusal(question):
        print(
            "\nMedIntel-AI: I cannot diagnose diseases or prescribe "
            "personalized treatment. Please consult a qualified "
            "healthcare professional for medical evaluation."
        )
        continue

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Return both input IDs and attention mask.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    print("\nMedIntel-AI:", response)
