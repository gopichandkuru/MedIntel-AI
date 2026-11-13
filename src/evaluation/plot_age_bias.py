
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "evaluation" / "age_bias_analysis.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "evaluation" / "age_bias_comparison.png"
)

df = pd.read_csv(INPUT_FILE)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "False_Positive_Rate"
]

plt.figure(figsize=(11, 6))

for metric in metrics:
    plt.plot(
        df["Age_Group"],
        df[metric],
        marker="o",
        label=metric
    )

plt.title("XGBoost Performance Across Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print("Age comparison graph saved:")
print(OUTPUT_FILE)
