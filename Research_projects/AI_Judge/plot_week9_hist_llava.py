import json
import matplotlib.pyplot as plt

hs = []
with open("runs/week8/llava_coco100.jsonl","r") as f:
    for line in f:
        r = json.loads(line)
        hs.append(r.get("hallucination") or 0.0)

plt.figure(figsize=(6,4))
plt.hist(hs, bins=[0,0.01,0.25,0.5,0.75,1.0])
plt.title("LLaVA Hallucination Score Distribution (COCO-100)")
plt.xlabel("Hallucination score")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig("outputs/week9_llava_hall_hist.png")
print("Saved: outputs/week9_llava_hall_hist.png")
