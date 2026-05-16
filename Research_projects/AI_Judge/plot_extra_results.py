import matplotlib.pyplot as plt

# =========================
# TOP VIOLATION CLAIMS
# =========================

claims = [
    "shirt",
    "wall",
    "table",
    "person",
    "window",
    "floor",
    "car",
    "chair",
    "truck",
    "flower"
]

counts = [51, 51, 43, 39, 31, 21, 18, 9, 9, 9]

plt.figure(figsize=(10,6))

plt.barh(claims, counts)

plt.xlabel("Count")
plt.ylabel("Unsupported Claim")
plt.title("Top Unsupported Claims")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("top_violations.png", dpi=300)

print("Saved: top_violations.png")


# =========================
# CLAIM COVERAGE PIE CHART
# =========================

covered = 422
not_covered = 53

labels = ["Captions with Claims", "No Claims"]

sizes = [covered, not_covered]

plt.figure(figsize=(7,7))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

plt.title("Claim Coverage on COCO Benchmark")

plt.savefig("claim_coverage_pie.png", dpi=300)

print("Saved: claim_coverage_pie.png")
