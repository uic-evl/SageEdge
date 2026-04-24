from __future__ import annotations
import re
from typing import Dict, Any, List, Set

STOP: Set[str] = {
    "are", "is", "was", "were", "be", "been", "being",
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "with", "for", "from",
    "across", "out", "some", "all", "they", "their", "them", "this", "that", "these", "those",
    "as", "into", "over", "under", "further", "closer", "together", "apart",
    "appear", "appears", "appeared", "enjoy", "enjoying", "enjoyed",
    "creating", "create", "created", "convey", "conveys", "conveyed",
    "engage", "engaging", "engaged",
    # common caption fluff
    "happily", "pleasant", "sunny", "vibrant", "lush", "various",
    "close", "closer", "background", "foreground",
    # common non-object words
    "room", "scene", "image", "picture", "photo",
    "there", "here", "next", "front", "top", "side",
    "standing", "sitting", "wearing", "surrounded",
    "visible", "clearly", "warm", "inviting",
}

COLORS: Set[str] = {
    "red", "blue", "green", "yellow", "black", "white",
    "gray", "grey", "brown", "orange", "pink", "purple"
}

COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

OBJECT_SYNONYMS = {
    "child": "person",
    "children": "person",
    "kid": "person",
    "kids": "person",
    "boy": "person",
    "girl": "person",
    "man": "person",
    "woman": "person",
    "people": "person",
    "men": "person",
    "women": "person",

    "bike": "bicycle",
    "bikes": "bicycle",
    "bicycles": "bicycle",

    "cars": "car",
    "dogs": "dog",
    "cats": "cat",
    "birds": "bird",
    "horses": "horse",
    "trucks": "truck",
    "buses": "bus",
    "motorcycles": "motorcycle",

    "chairs": "chair",
    "tables": "table",
    "tvs": "tv",
    "television": "tv",
    "televisions": "tv",
    "clocks": "clock",
    "vases": "vase",
    "plants": "plant",
    "flowers": "flower",
    "refrigerators": "refrigerator",
    "curtains": "curtain",
    "windows": "window",
    "dressers": "dresser",
    "shirts": "shirt",
    "jeans": "jeans",
    "floors": "floor",
    "walls": "wall",
}

OBJECT_VOCAB: Set[str] = {
    "person", "bicycle", "car", "bus", "truck", "motorcycle",
    "dog", "cat", "bird", "horse",
    "tree", "park", "field", "street", "road",

    "chair", "table", "tv", "clock", "vase", "plant", "refrigerator",
    "flower", "curtain", "window", "dresser", "shirt", "jeans", "floor", "wall",

    # keep these too in case they appear directly
    "dining table", "potted plant", "teddy bear", "stop sign", "cell phone",
}

MULTI_WORD_OBJECTS = {
    "dining tables": "dining table",
    "dining table": "dining table",

    "potted plants": "potted plant",
    "potted plant": "potted plant",

    "teddy bears": "teddy bear",
    "teddy bear": "teddy bear",

    "stop signs": "stop sign",
    "stop sign": "stop sign",

    "cell phones": "cell phone",
    "cell phone": "cell phone",
}

def normalize_token(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9_ -]", "", t)
    return t

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def parse_caption(caption: str) -> Dict[str, Any]:
    text = normalize_token(caption)

    objects: List[str] = []
    raw_objects: List[str] = []

    # detect multi-word objects first
    for phrase, canonical in sorted(MULTI_WORD_OBJECTS.items(), key=lambda x: len(x[0]), reverse=True):
        if phrase in text:
            raw_objects.append(phrase)
            objects.append(canonical)
            text = text.replace(phrase, " ")

    tokens = [tok for tok in re.split(r"\s+", text) if tok]
    tokens = [t for t in tokens if len(t) > 1]

    colors = sorted({t for t in tokens if t in COLORS})

    counts: List[Dict[str, Any]] = []
    for t in tokens:
        if t in COUNT_WORDS:
            counts.append({"word": t, "value": COUNT_WORDS[t]})

    for t in tokens:
        if t in STOP:
            continue
        if t in COLORS:
            continue
        if t in COUNT_WORDS:
            continue

        t2 = OBJECT_SYNONYMS.get(t, t)

        if t2 in STOP:
            continue

        if t2.endswith("s") and t2[:-1] in OBJECT_VOCAB:
            t2 = t2[:-1]

        if t2 in OBJECT_VOCAB:
            raw_objects.append(t)
            objects.append(t2)

    objects_unique = dedupe_keep_order(objects)
    raw_objects_unique = dedupe_keep_order(raw_objects)

    # remove weaker duplicates if multi-word form exists
    if "dining table" in objects_unique and "table" in objects_unique:
        objects_unique = [o for o in objects_unique if o != "table"]
    if "potted plant" in objects_unique and "plant" in objects_unique:
        objects_unique = [o for o in objects_unique if o != "plant"]
    if "tv" in objects_unique and "television" in objects_unique:
        objects_unique = [o for o in objects_unique if o != "television"]

    return {
        "objects": objects_unique,
        "raw_objects": raw_objects_unique,
        "colors": colors,
        "counts": counts,
        "raw": caption,
    }