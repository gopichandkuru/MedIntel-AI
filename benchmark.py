import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_CASES = [
    ("What are the symptoms of Type 2 diabetes?", ["thirst", "urination", "fatigue", "blurred", "blood sugar"]),
    ("What causes hypertension?", ["blood pressure", "salt", "sodium", "obesity", "stress", "arteries"]),
    ("How is asthma treated?", ["inhaler", "bronchodilator", "corticosteroid", "airway", "breathing"]),
    ("What is Sjogren's syndrome?", ["autoimmune", "dry eyes", "dry mouth", "salivary", "glands"]),
    ("What are common symptoms of anemia?", ["fatigue", "weakness", "pale", "shortness of breath", "red blood"]),
    ("What causes pneumonia?", ["infection", "bacteria", "virus", "lungs", "air sacs"]),
    ("What are the warning signs of a stroke?", ["face", "arm", "speech", "numbness", "brain"]),
    ("How is hypothyroidism treated?", ["levothyroxine", "hormone", "thyroid", "medication"]),
    ("What are the symptoms of a migraine?", ["headache", "nausea", "light", "sound", "throbbing"]),
    ("What is GERD?", ["acid", "reflux", "stomach", "esophagus", "heartburn"]),
]

SYSTEM_PROMPT = "You are MedIntel-AI, a concise medical Q&A assistant."

def evaluate_model(model_path, label):
    print(f"\nBenchmarking {label} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Warmup run
    warm_in = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**warm_in, max_new_tokens=10)

    total_tokens = 0
    total_time = 0.0
    scores = []

    for question, keywords in TEST_CASES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.2,
                do_sample=True,
                repetition_penalty=1.15,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        gen_ids = out[0][inputs.input_ids.shape[1]:]
        num_tokens = len(gen_ids)
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()

        hits = sum(1 for kw in keywords if kw in response)
        scores.append((hits / len(keywords)) * 100)
        total_tokens += num_tokens
        total_time += elapsed

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "accuracy": sum(scores) / len(scores),
        "avg_tokens": total_tokens / len(TEST_CASES),
        "speed": total_tokens / total_time,
    }

base_res = evaluate_model("Qwen/Qwen2.5-0.5B-Instruct", "Base Qwen2.5-0.5B")
ft_res = evaluate_model("models/medquad_merged", "MedIntel-AI (Fine-Tuned)")

print("\n--- Results ---")
print(f"Base Model  -> Accuracy: {base_res['accuracy']:.1f}% | Avg Tokens: {base_res['avg_tokens']:.1f} | Speed: {base_res['speed']:.1f} tok/s")
print(f"MedIntel-AI -> Accuracy: {ft_res['accuracy']:.1f}% | Avg Tokens: {ft_res['avg_tokens']:.1f} | Speed: {ft_res['speed']:.1f} tok/s")

# Plot PNG
labels = ["Base\nQwen2.5-0.5B", "MedIntel-AI\n(Fine-Tuned)"]
colors = ["#4A90E2", "#2ECC71"]
metrics = [
    ("Medical Accuracy (%)", [base_res["accuracy"], ft_res["accuracy"]], "{:.1f}%"),
    ("Avg Output Tokens", [base_res["avg_tokens"], ft_res["avg_tokens"]], "{:.1f}"),
    ("Inference Speed (tok/s)", [base_res["speed"], ft_res["speed"]], "{:.1f}"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=300)
fig.suptitle("Base Qwen2.5-0.5B vs MedIntel-AI Benchmark", fontsize=14, fontweight="bold")

for ax, (title, vals, fmt) in zip(axes, metrics):
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (max(vals) * 0.03),
                fmt.format(v), ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("benchmark_comparison.png", dpi=300)
print("\nSaved chart to benchmark_comparison.png")
