import json
import matplotlib.pyplot as plt

with open("outputs/week9_violation_report.json") as f:
    data = json.load(f)

top = data["llava"]["top_claims"]
labels = [x[0] for x in top]
values = [x[1] for x in top]

plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.title("Most Frequent Unsupported Claims (LLaVA)")
plt.xlabel("Claim")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/week9_top_unsupported_claims.png")
print("Saved: outputs/week9_top_unsupported_claims.png")
