# Import Unsloth before Transformers and TRL
import unsloth

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = "data/processed/medquad/sft/train_sft.jsonl"
OUTPUT_DIR = "models/medintel-smoke-test"

print("\n========== MEDINTEL AI SMOKE TEST ==========")
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Load a small training subset
dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE}
)["train"]

dataset = dataset.select(range(min(64, len(dataset))))

print("Smoke-test examples:", len(dataset))

# Load quantized model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

# Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=1024,
)

print("\nLoRA adapters attached successfully.")

# Training configuration
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    max_steps=5,
    learning_rate=1e-4,
    warmup_steps=0,

    logging_steps=1,
    eval_strategy="no",
    save_strategy="no",

    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",

    fp16=False,
    bf16=True,

    gradient_checkpointing=True,

    dataset_text_field="text",
    max_length=1024,
    packing=False,

    seed=3407,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

print("\nStarting 5 training steps...")
trainer.train()

print("\nSaving smoke-test adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n========== SMOKE TEST COMPLETED ==========")
print("Output:", OUTPUT_DIR)
print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)
print(
    "GPU memory reserved:",
    round(torch.cuda.memory_reserved() / 1024**3, 2),
    "GB"
)
