import json
from collections import Counter

path = "runs/test_run/predictions.jsonl"

n = 0
hall_sum = 0.0
ground_sum = 0.0
viol_sum = 0
viol_claims = Counter()

with open(path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        hall_sum += float(r.get("hallucination", 0.0) or 0.0)
        ground_sum += float(r.get("grounding", 0.0) or 0.0)

        v = r.get("violations", []) or []
        viol_sum += len(v)
        for item in v:
            c = item.get("claim")
            if c:
                viol_claims[c] += 1

print("N =", n)
print("avg_hallucination =", hall_sum / n if n else 0.0)
print("avg_grounding =", ground_sum / n if n else 0.0)
print("avg_num_violations =", viol_sum / n if n else 0.0)
print("top_violation_claims =", viol_claims.most_common(10))
