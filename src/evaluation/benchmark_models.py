
import csv
import gc
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = Path("models/medquad_qlora_final")
TEST_FILE = Path("data/processed/medquad/test.jsonl")

OUTPUT_DIR = Path("results/benchmark")

NUM_QUESTIONS = 50
MAX_NEW_TOKENS = 160
MAX_INPUT_TOKENS = 1024
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    raise RuntimeError(
        "CUDA GPU is required for this 4-bit benchmark. "
        "Check your PyTorch CUDA installation."
    )


# ============================================================
# METRICS
# ============================================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(
        normalize_text(prediction) == normalize_text(reference)
    )


def token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = {}

    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0

    for token in ref_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction, reference):
    pred = normalize_text(prediction).split()
    ref = normalize_text(reference).split()

    if not pred or not ref:
        return 0.0

    previous = [0] * (len(ref) + 1)

    for pred_token in pred:
        current = [0]

        for j, ref_token in enumerate(ref, start=1):
            if pred_token == ref_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))

        previous = current

    lcs = previous[-1]

    precision = lcs / len(pred)
    recall = lcs / len(ref)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ============================================================
# DATASET
# ============================================================

def load_test_data():
    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    records = []

    with TEST_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)

            question = item.get("question")
            answer = item.get("answer")

            if question and answer:
                records.append({
                    "id": item.get("id", ""),
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "source": item.get("source", ""),
                })

    if not records:
        raise ValueError("No valid question-answer records found.")

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    selected = records[:min(NUM_QUESTIONS, len(records))]

    print(f"Total test records: {len(records)}")
    print(f"Selected benchmark records: {len(selected)}")

    return selected


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(use_adapter):
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model in 4-bit...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if use_adapter:
        print("Loading fine-tuned QLoRA adapter...")

        model = PeftModel.from_pretrained(
            model,
            str(ADAPTER_PATH),
        )

    model.eval()

    return tokenizer, model


# ============================================================
# GENERATION
# ============================================================

def generate_answer(model, tokenizer, question):

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    input_device = model.get_input_embeddings().weight.device
    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    input_tokens = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    generated_tokens = max(
        0,
        output.shape[1] - input_tokens,
    )

    answer = tokenizer.decode(
        output[0][input_tokens:],
        skip_special_tokens=True,
    ).strip()

    speed = (
        generated_tokens / elapsed
        if elapsed > 0 else 0
    )

    return answer, elapsed, generated_tokens, speed


# ============================================================
# BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(model_name, use_adapter, test_data):

    print("\n" + "=" * 65)
    print(f"BENCHMARK: {model_name}")
    print("=" * 65)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer, model = load_model(use_adapter)

    results = []

    # Warm-up: excluded from measured results.
    print("\nRunning warm-up...")
    generate_answer(
        model,
        tokenizer,
        "What is diabetes?",
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for index, item in enumerate(test_data, start=1):

        print(
            f"\nQuestion {index}/{len(test_data)}: "
            f"{item['question'][:75]}"
        )

        prediction, latency, tokens, speed = generate_answer(
            model,
            tokenizer,
            item["question"],
        )

        reference = item["answer"]

        metrics = {
            "id": item["id"],
            "question": item["question"],
            "reference": reference,
            "prediction": prediction,
            "source": item["source"],
            "latency_seconds": latency,
            "generated_tokens": tokens,
            "tokens_per_second": speed,
            "exact_match": exact_match(prediction, reference),
            "token_f1": token_f1(prediction, reference),
            "rouge_l": rouge_l(prediction, reference),
        }

        results.append(metrics)

        print(f"Latency: {latency:.3f} seconds")
        print(f"Generation speed: {speed:.2f} tokens/sec")
        print(f"Token F1: {metrics['token_f1']:.4f}")

    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    summary = {
        "model": model_name,
        "questions_tested": len(results),
        "avg_latency_seconds": sum(
            x["latency_seconds"] for x in results
        ) / len(results),
        "avg_tokens_per_second": sum(
            x["tokens_per_second"] for x in results
        ) / len(results),
        "exact_match_percent": 100 * sum(
            x["exact_match"] for x in results
        ) / len(results),
        "token_f1_percent": 100 * sum(
            x["token_f1"] for x in results
        ) / len(results),
        "rouge_l_percent": 100 * sum(
            x["rouge_l"] for x in results
        ) / len(results),
        "peak_gpu_memory_gb": peak_memory_gb,
    }

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    return summary, results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(summaries, detailed_results):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "comparison_summary.json"
    details_path = OUTPUT_DIR / "detailed_results.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=4)

    with details_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        if detailed_results:
            writer = csv.DictWriter(
                file,
                fieldnames=detailed_results[0].keys(),
            )

            writer.writeheader()
            writer.writerows(detailed_results)

    print("\nResults saved:")
    print(summary_path)
    print(details_path)


# ============================================================
# FINAL COMPARISON
# ============================================================

def print_comparison(base, tuned):

    print("\n")
    print("=" * 85)
    print("MEDINTEL-AI: BASE MODEL VS QLoRA FINE-TUNED MODEL")
    print("=" * 85)

    columns = [
        ("Average latency (seconds)", "avg_latency_seconds"),
        ("Generation speed (tokens/sec)", "avg_tokens_per_second"),
        ("Exact Match (%)", "exact_match_percent"),
        ("Token F1 (%)", "token_f1_percent"),
        ("ROUGE-L (%)", "rouge_l_percent"),
        ("Peak GPU memory (GB)", "peak_gpu_memory_gb"),
    ]

    print(f"\n{'Metric':<35} {'Base Model':>20} {'Fine-tuned':>20}")
    print("-" * 80)

    for label, key in columns:
        print(
            f"{label:<35} "
            f"{base[key]:>20.3f} "
            f"{tuned[key]:>20.3f}"
        )

    print("\nPercentage change (fine-tuned vs base):")

    for label, key in columns:
        base_value = base[key]
        tuned_value = tuned[key]

        if base_value:
            change = (
                (tuned_value - base_value) / abs(base_value)
            ) * 100

            print(f"{label}: {change:+.2f}%")

    print("\nNote:")
    print("- Lower latency is better.")
    print("- Higher generation speed is better.")
    print("- Higher EM, F1 and ROUGE-L are better.")
    print("- Lower GPU memory usage is generally better.")
    print("- These are text-overlap metrics, not clinical accuracy.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MEDINTEL-AI MODEL BENCHMARK")
    print("=" * 65)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Adapter not found: {ADAPTER_PATH}"
        )

    test_data = load_test_data()

    # Run base model first.
    base_summary, base_results = benchmark_model(
        "Qwen2.5-0.5B-Instruct (Base)",
        False,
        test_data,
    )

    # Release base model before loading adapter.
    tuned_summary, tuned_results = benchmark_model(
        "Qwen2.5-0.5B-Instruct + MedIntel QLoRA",
        True,
        test_data,
    )

    summaries = {
        "base_model": base_summary,
        "fine_tuned_model": tuned_summary,
        "benchmark_settings": {
            "test_questions": len(test_data),
            "quantization": "4-bit NF4",
            "max_new_tokens": MAX_NEW_TOKENS,
            "random_seed": RANDOM_SEED,
            "dataset": str(TEST_FILE),
        },
    }

    detailed_results = []

    for base, tuned in zip(base_results, tuned_results):
        detailed_results.append({
            "id": base["id"],
            "question": base["question"],
            "reference": base["reference"],
            "source": base["source"],
            "base_prediction": base["prediction"],
            "tuned_prediction": tuned["prediction"],
            "base_latency": base["latency_seconds"],
            "tuned_latency": tuned["latency_seconds"],
            "base_tokens_per_second": base["tokens_per_second"],
            "tuned_tokens_per_second": tuned["tokens_per_second"],
            "base_exact_match": base["exact_match"],
            "tuned_exact_match": tuned["exact_match"],
            "base_token_f1": base["token_f1"],
            "tuned_token_f1": tuned["token_f1"],
            "base_rouge_l": base["rouge_l"],
            "tuned_rouge_l": tuned["rouge_l"],
        })

    save_results(summaries, detailed_results)
    print_comparison(base_summary, tuned_summary)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
