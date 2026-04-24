import json, argparse
from collections import Counter

def load(path):
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else 0.0

def summarize(path):
    rows=list(load(path))
    n=len(rows)
    claims=[len(r.get("claims") or []) for r in rows]
    covered=[r for r,c in zip(rows,claims) if c>0]
    hall_all=mean((r.get("hallucination") or 0.0) for r in rows)
    grd_all=mean((r.get("grounding") or 0.0) for r in rows)
    hall_cov=mean((r.get("hallucination") or 0.0) for r in covered)
    grd_cov=mean((r.get("grounding") or 0.0) for r in covered)

    vio=Counter()
    for r in rows:
        for v in (r.get("violations") or []):
            if isinstance(v, dict) and v.get("type")=="object_not_detected":
                vio[v.get("claim","")] += 1

    return {
        "N": n,
        "claim_coverage": f"{len(covered)}/{n}",
        "avg_claims": mean(claims),
        "hall_all": hall_all,
        "grd_all": grd_all,
        "hall_cov": hall_cov,
        "grd_cov": grd_cov,
        "top_violations": vio.most_common(10)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--moondream", required=True)
    ap.add_argument("--llava", required=True)
    args=ap.parse_args()

    m=summarize(args.moondream)
    l=summarize(args.llava)

    print("\nMOONDREAM:", m)
    print("\nLLAVA:", l)

    # quick side-by-side table
    print("\n=== SIDE BY SIDE ===")
    print("metric                moondream            llava")
    print(f"N                     {m['N']:>7}            {l['N']:>7}")
    print(f"claim_coverage        {m['claim_coverage']:>7}            {l['claim_coverage']:>7}")
    print(f"avg_claims            {m['avg_claims']:.2f}               {l['avg_claims']:.2f}")
    print(f"hall_all              {m['hall_all']:.4f}             {l['hall_all']:.4f}")
    print(f"grd_all               {m['grd_all']:.4f}             {l['grd_all']:.4f}")
    print(f"hall_cov              {m['hall_cov']:.4f}             {l['hall_cov']:.4f}")
    print(f"grd_cov               {m['grd_cov']:.4f}             {l['grd_cov']:.4f}")

if __name__=="__main__":
    main()
