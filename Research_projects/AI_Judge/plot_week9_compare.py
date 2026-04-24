import json
import matplotlib.pyplot as plt

# paste your exact numbers (from compare_runs.py)
moondream = {
  "claim_coverage": 67,
  "hall_cov": 0.0,
  "grd_cov": 1.0
}
llava = {
  "claim_coverage": 57,
  "hall_cov": 0.04970760233918129,
  "grd_cov": 0.8625730994152047
}

labels = ["Claim coverage (%)", "Hallucination (>=1 claim)", "Grounding (>=1 claim)"]
m_vals = [moondream["claim_coverage"], moondream["hall_cov"], moondream["grd_cov"]]
l_vals = [llava["claim_coverage"], llava["hall_cov"], llava["grd_cov"]]

x = range(len(labels))
w = 0.35

plt.figure(figsize=(7,4))
plt.bar([i - w/2 for i in x], m_vals, width=w, label="Moondream2")
plt.bar([i + w/2 for i in x], l_vals, width=w, label="LLaVA")

plt.xticks(list(x), labels, rotation=15, ha="right")
plt.ylabel("Value")
plt.title("AI Judge: Moondream2 vs LLaVA (COCO-100)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/week9_compare_plot.png")
print("Saved: outputs/week9_compare_plot.png")
