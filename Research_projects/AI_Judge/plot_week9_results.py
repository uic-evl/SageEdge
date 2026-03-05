import json
import matplotlib.pyplot as plt

with open("outputs/week9_violation_report.json") as f:
    data = json.load(f)

counts = data["llava"]["category_counts"]

labels = list(counts.keys())
values = list(counts.values())

plt.figure(figsize=(6,4))
plt.bar(labels, values)

plt.title("Hallucination Causes (LLaVA)")
plt.ylabel("Count")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("outputs/week9_violation_plot.png")

print("Saved: outputs/week9_violation_plot.png")
