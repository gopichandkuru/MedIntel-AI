
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTConfig, SFTTrainer

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = "data/processed/medquad/sft/train_sft.jsonl"
VALID_FILE = "data/processed/medquad/sft/valid_sft.jsonl"

OUTPUT_DIR = "models/medquad_qlora"
FINAL_DIR = "models/medquad_qlora_final"

MAX_LENGTH = 1024

print("=" * 55)
print("       MEDINTEL-AI FULL QLoRA TRAINING")
print("=" * 55)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available!")

print("GPU:", torch.cuda.get_device_name(0))

# ==========================================
# 2. LOAD DATASETS
# ==========================================

print("\nLoading MedQuAD datasets...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALID_FILE,
    },
)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

for split_name, split in [
    ("Training", train_dataset),
    ("Validation", eval_dataset),
]:
    if "text" not in split.column_names:
        raise ValueError(
            f"{split_name} dataset must contain a 'text' column. "
            f"Found: {split.column_names}"
        )

print("Training examples:", len(train_dataset))
print("Validation examples:", len(eval_dataset))

# ==========================================
# 3. LOAD TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 4. 4-BIT QUANTIZATION
# ==========================================

print("\nConfiguring 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ==========================================
# 5. LOAD BASE MODEL
# ==========================================

print("\nLoading Qwen2.5-0.5B model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# ==========================================
# 6. PREPARE QLoRA
# ==========================================

print("\nPreparing model for QLoRA...")

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ==========================================
# 7. TRAINING CONFIGURATION
# ==========================================

print("\nConfiguring training...")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    # Validation
    do_eval=True,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=200,
    eval_accumulation_steps=1,

    # Checkpoints
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,

    # Optimization
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    max_grad_norm=1.0,

    # Memory and precision
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    # Dataset
    dataset_text_field="text",
    max_length=MAX_LENGTH,
    packing=False,

    # Logging
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,
    report_to="none",

    # Other
    seed=42,
    data_seed=42,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    remove_unused_columns=False,

    # Keep best validation checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ==========================================
# 8. INITIALIZE TRAINER
# ==========================================

print("\nInitializing SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# ==========================================
# 9. START TRAINING
# ==========================================

print("\n" + "=" * 55)
print("STARTING FULL MEDQUAD QLoRA TRAINING")
print("=" * 55)

torch.cuda.empty_cache()

train_result = trainer.train()

print("\nTraining completed!")

# ==========================================
# 10. FINAL EVALUATION
# ==========================================

print("\nRunning final validation...")

eval_metrics = trainer.evaluate()

print("Final validation metrics:", eval_metrics)

# ==========================================
# 11. SAVE FINAL ADAPTER
# ==========================================

print("\nSaving final LoRA adapter...")

os.makedirs(FINAL_DIR, exist_ok=True)

trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

# Save metrics
trainer.save_metrics("train", train_result.metrics)
trainer.save_metrics("eval", eval_metrics)
trainer.save_state()

print("\n" + "=" * 55)
print("MEDINTEL-AI QLoRA TRAINING COMPLETED!")
print("=" * 55)

print("Final model directory:", FINAL_DIR)
print("Training loss:", train_result.metrics.get("train_loss"))
print("Validation loss:", eval_metrics.get("eval_loss"))

print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)
