from __future__ import annotations
import re
from typing import Dict, Any, List, Set

STOP: Set[str] = {
    "are","is","was","were","be","been","being",
    "a","an","the","and","or","of","to","in","on","at","with","for","from",
    "across","out","some","all","they","their","them","this","that","these","those",
    "as","into","over","under","further","closer","together","apart",
    "appear","appears","appeared","enjoy","enjoying","enjoyed",
    "creating","create","created","convey","conveys","conveyed","engage","engaging","engaged",
    # common caption fluff
    "happily","pleasant","sunny","vibrant","lush","various","close","closer","background","foreground"
}
# Minimal vocab (extend later)
COLORS: Set[str] = {"red","blue","green","yellow","black","white","gray","grey","brown","orange","pink","purple"}
COUNT_WORDS = {
    "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9, "ten":10
}

# A small starter set; you can expand / replace with COCO classes later
OBJECT_SYNONYMS = {
    "child": "person",
    "children": "person",
    "kid": "person",
    "kids": "person",
    "boy": "person",
    "girl": "person",
    "bike":"bicycle",
    "bikes":"bicycle",
    "man":"person",
    "woman":"person",
    "people":"person",
    "men":"person",
    "women":"person",
    "kids":"person",
    "kid":"person",
    "cars":"car",
    "dogs":"dog",
    "bicycles":"bicycle",
}
OBJECT_VOCAB: Set[str] = {
    # start small; expand later (or replace with COCO classes)
    "person","child","children","man","woman","boy","girl",
    "bicycle","car","bus","truck","motorcycle",
    "dog","cat","bird","horse",
    "tree","trees","park","field"
    "car","cars",
    "dog","dogs",
    "bicycle","bicycles",
    "street","road"
}
def normalize_token(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9_ -]", "", t)
    return t

def parse_caption(caption: str) -> Dict[str, Any]:
    text = normalize_token(caption)
    tokens = [tok for tok in re.split(r"\s+", text) if tok]

    colors = sorted({t for t in tokens if t in COLORS})

    counts = []
    for t in tokens:
        if t in COUNT_WORDS:
            counts.append({"word": t, "value": COUNT_WORDS[t]})

    # naive object extraction: scan tokens and map synonyms
    objects: List[str] = []
    for t in tokens:
        if t in STOP:
            continue

        t2 = OBJECT_SYNONYMS.get(t, t)

        # only keep if it looks like a real object word
        if t2 in COLORS or t2 in COUNT_WORDS:
            continue

        if t2 in OBJECT_VOCAB:
            objects.append(t2)

    # de-dup but preserve order
    seen = set()
    objects_unique = []
    for o in objects:
        if o not in seen:
            seen.add(o)
            objects_unique.append(o)

    return {
        "objects": objects_unique,
        "colors": colors,
        "counts": counts,
        "raw": caption,
    }
