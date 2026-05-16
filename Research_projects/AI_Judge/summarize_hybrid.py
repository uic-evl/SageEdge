import json

path = "runs/week14_hybrid_500.jsonl"

rule_h = []
rule_g = []

ret_h = []
ret_g = []

hyb_h = []
hyb_g = []

count = 0

with open(path) as f:
    for line in f:
        d = json.loads(line)

        # rule-only
        if "rule_hallucination" in d:
            rule_h.append(d["rule_hallucination"])

        if "rule_grounding" in d:
            rule_g.append(d["rule_grounding"])

        # retrieval-only
        if "retrieval_hallucination" in d:
            ret_h.append(d["retrieval_hallucination"])

        if "retrieval_grounding" in d:
            ret_g.append(d["retrieval_grounding"])

        # hybrid final
        if "hallucination" in d:
            hyb_h.append(d["hallucination"])

        if "grounding" in d:
            hyb_g.append(d["grounding"])

        count += 1

print("\n========== HYBRID AI JUDGE RESULTS ==========\n")

print(f"Total Samples: {count}")

print("\n--- RULE ONLY ---")
print("Avg Hallucination:", round(sum(rule_h)/len(rule_h), 4))
print("Avg Grounding:", round(sum(rule_g)/len(rule_g), 4))

print("\n--- RETRIEVAL ONLY ---")
print("Avg Hallucination:", round(sum(ret_h)/len(ret_h), 4))
print("Avg Grounding:", round(sum(ret_g)/len(ret_g), 4))

print("\n--- HYBRID FINAL ---")
print("Avg Hallucination:", round(sum(hyb_h)/len(hyb_h), 4))
print("Avg Grounding:", round(sum(hyb_g)/len(hyb_g), 4))

print("\n=============================================\n")
