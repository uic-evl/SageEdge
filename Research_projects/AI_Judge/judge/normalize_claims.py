from __future__ import annotations
import re
from typing import Set, List

CANONICAL_OBJECTS: Set[str] = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

NON_DETECTABLE = {
    "tree", "trees", "park", "field", "street", "road", "kitchen", "sky", "grass"
}

SYNONYM_MAP = {
    "man": "person",
    "bears": "bear",
    "teddy": "teddy bear",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "child": "person",
    "children": "person",
    "kid": "person",
    "kids": "person",
    "people": "person",
    "men": "person",
    "women": "person",

    "bike": "bicycle",
    "bikes": "bicycle",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "cycle": "bicycle",

    "car": "car",
    "cars": "car",
    "automobile": "car",
    "automobiles": "car",
    "vehicle": "car",
    "vehicles": "car",

    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",

    "dogs": "dog",
    "cats": "cat",
    "birds": "bird",
    "horses": "horse",

    "sofa": "couch",
    "sofas": "couch",
    "television": "tv",
    "televisions": "tv",
    "tv monitor": "tv",
    "cellphone": "cell phone",
    "cellphones": "cell phone",
    "phone": "cell phone",
    "phones": "cell phone",
    "mobile": "cell phone",
    "mobile phone": "cell phone",
    "fridge": "refrigerator",
    "fridges": "refrigerator",
    "table": "dining table",

    "traffic signal": "traffic light",
}

STOPWORDS = {
    "are","is","was","were","be","been","being",
    "a","an","the","and","or","of","to","in","on","at","with","for","from",
    "across","out","some","all","they","their","them","this","that","these","those",
    "as","into","over","under","further","closer","together","apart",
    "appear","appears","appeared","enjoy","enjoying","enjoyed",
    "creating","create","created","convey","conveys","conveyed","engage","engaging","engaged",
    "happily","pleasant","sunny","vibrant","lush","various","close","closer","background","foreground",
    "photo","image","objects","object","picture","scene","detail","details","including",
    "activity","activities"
}

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_label(label: str) -> str:
    label = clean_text(label)
    return SYNONYM_MAP.get(label, label)

def normalize_detection_label(label: str) -> str:
    return normalize_label(label)

def is_canonical_or_nondetectable(label: str) -> bool:
    return label in CANONICAL_OBJECTS or label in NON_DETECTABLE

def normalize_claim_list(labels: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for label in labels:
        norm = normalize_label(label)
        if not norm:
            continue
        if norm in STOPWORDS:
            continue
        if not is_canonical_or_nondetectable(norm):
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out
