
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("CUDA:", torch.cuda.is_available())

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load a small training sample
records = []
with open("data/processed/medquad/sft/train_sft.jsonl", "r") as f:
    for line in f:
        records.append(json.loads(line))
        if len(records) == 32:
            break

dataset = Dataset.from_list(records)
print("Training samples:", len(dataset))

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto",
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training configuration
args = TrainingArguments(
    output_dir="models/qlora_smoke_test",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=5,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    optim="adamw_torch",
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("\nStarting 5-step QLoRA training test...")
trainer.train()

print("\nQLoRA TRAINING SMOKE TEST: SUCCESS")
print("Final loss:", trainer.state.log_history[-1].get("loss"))
