import matplotlib.pyplot as plt
import numpy as np

methods = ["Rule", "Retrieval", "Hybrid"]

hallucination = [0.339, 0.377, 0.342]
grounding = [0.491, 0.490, 0.491]

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(9,6))

bars1 = plt.bar(
    x - width/2,
    hallucination,
    width,
    label='Hallucination'
)

bars2 = plt.bar(
    x + width/2,
    grounding,
    width,
    label='Grounding'
)

plt.xticks(x, methods)

plt.ylabel("Score")
plt.xlabel("Method")

plt.title("Hybrid AI Judge Evaluation on COCO")

plt.ylim(0, 0.6)

plt.legend()

# value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            f"{height:.3f}",
            ha='center',
            fontsize=9
        )

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig("final_hybrid_comparison_v2.png", dpi=300)

print("Saved: final_hybrid_comparison_v2.png")
