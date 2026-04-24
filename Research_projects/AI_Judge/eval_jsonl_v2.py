import json
import argparse
from collections import Counter

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def mean(xs):
    xs = list(xs)
    return sum(xs)/len(xs) if xs else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    args = p.parse_args()

    rows = list(load_jsonl(args.inp))
    n = len(rows)

    claim_counts = [len(r.get("claims", []) or []) for r in rows]
    covered = [r for r,c in zip(rows, claim_counts) if c > 0]
    n_cov = len(covered)

    # Overall (as-is)
    avg_h = mean(r.get("hallucination", 0.0) or 0.0 for r in rows)
    avg_g = mean(r.get("grounding", 0.0) or 0.0 for r in rows)

    # Conditional on having >=1 claim (scientifically fair)
    avg_h_cov = mean(r.get("hallucination", 0.0) or 0.0 for r in covered)
    avg_g_cov = mean(r.get("grounding", 0.0) or 0.0 for r in covered)

    # Top violation claim names (if your violations store claim strings)
    vc = Counter()
    for r in rows:
        for v in (r.get("violations") or []):
            if isinstance(v, str):
                vc[v] += 1
            elif isinstance(v, dict) and "claim" in v:
                vc[str(v["claim"])] += 1

    print(f"N = {n}")
    print(f"claim_coverage = {n_cov}/{n} ({(n_cov/n*100 if n else 0):.1f}%)")
    print(f"avg_claims_per_caption = {mean(claim_counts):.2f}")
    print(f"avg_hallucination_all = {avg_h}")
    print(f"avg_grounding_all = {avg_g}")

    print("\n(Only captions with >=1 claim)")
    print(f"avg_hallucination_cov = {avg_h_cov}")
    print(f"avg_grounding_cov = {avg_g_cov}")

    print("\nTop violation claims:", vc.most_common(10))

if __name__ == "__main__":
    main()
