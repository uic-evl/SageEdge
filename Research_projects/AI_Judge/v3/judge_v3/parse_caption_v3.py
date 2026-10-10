"""
parse_caption_v3 — object-claim extraction for the AI Judge.

Fixes over parse_caption_v2:
  1. Negation: objects inside "no ...", "without ...", "not ...", "nobody" etc.
     are NOT claims (e.g. "There are no people visible").
  2. Compound nouns that are NOT the COCO object are blocked before matching
     ("table lamp", "bus stop", "train car", "banana plant", "phone booth" ...).
  3. Much larger synonym map (people words, birds, planes, boats, balls, ...),
     based on the COCO category names and CHAIR-style synonym lists.
  4. Generic words ("vehicle", "animal") are no longer forced onto one class
     (vehicle used to become "car"); they are kept as out-of-vocabulary.

Output format is identical to v2: {"objects", "raw_objects", "colors", "counts", "raw"},
plus "negated" (objects that were mentioned under negation, for analysis).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------- vocabulary
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
COCO_SET = set(COCO_CLASSES)

# Words/phrases -> COCO class. Plurals are handled by singularize().
SYNONYMS: Dict[str, str] = {
    # person
    **{w: "person" for w in [
        "person", "people", "persons", "man", "men", "woman", "women", "boy", "girl", "child",
        "children", "kid", "kids", "baby", "babies", "toddler", "infant", "lady", "ladies",
        "guy", "gentleman", "teenager", "teen", "player", "surfer", "skier",
        "snowboarder", "skateboarder", "rider", "biker", "cyclist", "pedestrian", "batter",
        "pitcher", "catcher", "umpire", "fielder", "spectator", "crowd", "passenger",
        "tourist", "chef", "cook", "worker", "officer", "policeman", "policemen", "soldier",
        "athlete", "family", "couple", "individual", "someone", "friend", "student",
        "customer", "vendor", "driver", "bride", "groom", "daughter",
        "son", "brother", "sister",
    ]},
    # vehicles
    "bike": "bicycle", "bicycle": "bicycle",
    "motorbike": "motorcycle", "motorcycle": "motorcycle", "scooter": "motorcycle", "moped": "motorcycle",
    "car": "car", "automobile": "car", "taxi": "car", "cab": "car", "sedan": "car", "suv": "car", "jeep": "car",
    "plane": "airplane", "airplane": "airplane", "aeroplane": "airplane", "jet": "airplane",
    "airliner": "airplane", "aircraft": "airplane",
    "bus": "bus", "trolley": "bus",
    "train": "train", "locomotive": "train", "tram": "train", "subway": "train",
    "truck": "truck", "pickup": "truck", "lorry": "truck",
    "boat": "boat", "ship": "boat", "sailboat": "boat", "canoe": "boat", "kayak": "boat",
    "yacht": "boat", "ferry": "boat",
    # outdoor
    "traffic light": "traffic light", "traffic signal": "traffic light", "stoplight": "traffic light",
    "fire hydrant": "fire hydrant", "hydrant": "fire hydrant",
    "stop sign": "stop sign", "parking meter": "parking meter", "bench": "bench",
    # animals
    "bird": "bird", "duck": "bird", "goose": "bird", "geese": "bird", "seagull": "bird",
    "gull": "bird", "pigeon": "bird", "parrot": "bird", "swan": "bird", "hen": "bird", "rooster": "bird",
    "owl": "bird", "eagle": "bird", "crow": "bird",
    "cat": "cat", "kitten": "cat", "kitty": "cat",
    "dog": "dog", "puppy": "dog", "puppies": "dog",
    "horse": "horse", "pony": "horse", "foal": "horse",
    "sheep": "sheep", "lamb": "sheep", "ram": "sheep",
    "cow": "cow", "cattle": "cow", "calf": "cow", "calves": "cow", "bull": "cow", "ox": "cow", "oxen": "cow",
    "elephant": "elephant", "bear": "bear", "zebra": "zebra", "giraffe": "giraffe",
    # accessories
    "backpack": "backpack", "rucksack": "backpack",
    "umbrella": "umbrella", "parasol": "umbrella",
    "handbag": "handbag", "purse": "handbag",
    "tie": "tie", "necktie": "tie",
    "suitcase": "suitcase", "luggage": "suitcase",
    # sports
    "frisbee": "frisbee", "ski": "skis", "skis": "skis", "snowboard": "snowboard",
    "sports ball": "sports ball", "ball": "sports ball", "football": "sports ball",
    "soccer ball": "sports ball", "tennis ball": "sports ball", "basketball": "sports ball",
    "kite": "kite", "baseball bat": "baseball bat", "bat": "baseball bat",
    "baseball glove": "baseball glove", "mitt": "baseball glove",
    "skateboard": "skateboard", "surfboard": "surfboard",
    "tennis racket": "tennis racket", "racket": "tennis racket", "racquet": "tennis racket",
    # kitchen
    "bottle": "bottle", "wine glass": "wine glass", "cup": "cup", "mug": "cup",
    "fork": "fork", "knife": "knife", "knives": "knife", "spoon": "spoon", "bowl": "bowl",
    # food
    "banana": "banana", "apple": "apple", "sandwich": "sandwich", "burger": "sandwich",
    "hamburger": "sandwich", "sub": "sandwich", "orange": "orange", "broccoli": "broccoli",
    "carrot": "carrot", "hot dog": "hot dog", "hotdog": "hot dog", "pizza": "pizza",
    "donut": "donut", "doughnut": "donut", "cake": "cake", "cupcake": "cake",
    # furniture
    "chair": "chair", "stool": "chair", "couch": "couch", "sofa": "couch",
    "potted plant": "potted plant", "houseplant": "potted plant", "plant": "potted plant",
    "bed": "bed", "dining table": "dining table", "table": "dining table",
    "toilet": "toilet",
    # electronics
    "tv": "tv", "television": "tv", "laptop": "laptop", "computer mouse": "mouse",
    "mouse": "mouse", "remote": "remote", "remote control": "remote", "controller": "remote",
    "keyboard": "keyboard", "cell phone": "cell phone", "cellphone": "cell phone",
    "phone": "cell phone", "smartphone": "cell phone", "mobile phone": "cell phone",
    # appliances
    "microwave": "microwave", "oven": "oven", "stove": "oven", "toaster": "toaster",
    "sink": "sink", "refrigerator": "refrigerator", "fridge": "refrigerator",
    # indoor
    "book": "book", "clock": "clock", "vase": "vase", "scissors": "scissors",
    "teddy bear": "teddy bear", "teddy": "teddy bear", "stuffed animal": "teddy bear",
    "hair drier": "hair drier", "hair dryer": "hair drier", "blow dryer": "hair drier",
    "toothbrush": "toothbrush",
}

# Visual words YOLO cannot verify: kept for analysis as out-of-vocabulary claims.
OUT_OF_VOCAB = {
    "tree", "park", "field", "street", "road", "kitchen", "sky", "grass", "flower",
    "curtain", "window", "dresser", "shirt", "jeans", "floor", "wall", "room", "fireplace",
    "appliance", "building", "mountain", "beach", "ocean", "water", "snow", "sidewalk",
    "fence", "lamp", "mirror", "cabinet", "counter", "shelf", "door", "sign", "pole",
    "vehicle", "animal", "food", "plate", "hat", "helmet", "jacket", "bag", "glasses",
    "rug", "pillow", "blanket", "towel", "basket", "box", "desk",
}

# Compound phrases whose head word would otherwise be mis-mapped. Matched first and
# replaced by the given label (None = drop the phrase entirely, not a COCO claim).
BLOCKED_COMPOUNDS: Dict[str, str | None] = {
    "table lamp": "lamp", "table tennis": None, "table cloth": None, "tablecloth": None,
    "bus stop": None, "bus station": None, "bus shelter": None, "bus terminal": None, "bus lane": None,
    "train car": "train", "train cars": "train", "rail car": "train", "railcar": "train",
    "subway car": "train", "passenger car": "train", "cable car": None, "car park": None,
    "train station": None, "train track": None, "train tracks": None, "railroad track": None,
    "banana plant": None, "banana tree": "tree", "banana trees": "tree", "power plant": None,
    "phone booth": None, "phone line": None, "phone lines": None, "telephone pole": "pole",
    "ski lift": None, "ski slope": None, "ski resort": None, "ski pole": None, "ski poles": None,
    "tennis court": None, "baseball field": "field", "baseball game": None, "baseball diamond": None,
    "football field": "field", "football game": None, "soccer game": None,
    "baseball player": "person", "tennis player": "person", "soccer player": "person", "soccer field": "field", "dog park": "park", "bird feeder": None,
    "bird cage": None, "hot air balloon": None, "horse drawn": None, "cat food": None,
    "dog food": None, "bed sheet": None, "bed sheets": None, "flower bed": None,
    "clock tower": None, "book shelf": "shelf", "bookshelf": "shelf", "bookcase": "shelf",
    "fire truck": "truck", "fire engine": "truck", "police car": "car", "pickup truck": "truck",
    "orange juice": None, "apple logo": None, "cake stand": None, "pizza box": "box",
    "pizza cutter": None, "sink faucet": None, "chair lift": None, "wheelchair": None,
    "tv stand": None, "remote area": None, "remote location": None, "mouse pad": None,
    "man made": None, "man-made": None,
}

NEGATORS = {"no", "not", "without", "none", "nobody", "neither", "nor", "never",
            "absent", "lacking", "free", "empty"}
NO_PERSON_WORDS = {"nobody": "person", "noone": "person", "unoccupied": None, "unmanned": None}
CLAUSE_BREAKS = {"but", "while", "and", "though", "although", "however", "except", "yet"}
NEG_WINDOW = 6  # tokens after a negator that count as negated

COLORS = {"red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown",
          "orange", "pink", "purple"}
COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_IRREGULAR = {"people": "people", "men": "man", "women": "woman", "children": "child",
              "geese": "goose", "mice": "mouse", "knives": "knife", "calves": "calf",
              "oxen": "ox", "skis": "skis", "glasses": "glasses", "scissors": "scissors",
              "sheep": "sheep", "cattle": "cattle", "series": "series", "bus": "bus",
              "buses": "bus", "boxes": "box", "benches": "bench", "couches": "couch",
              "sandwiches": "sandwich", "dishes": "dish", "brushes": "brush",
              "toothbrushes": "toothbrush", "ladies": "lady", "babies": "baby",
              "puppies": "puppy", "teddies": "teddy", "ponies": "pony", "cherries": "cherry"}


def singularize(tok: str) -> str:
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ches", "shes", "sses", "xes")) and len(tok) > 4:
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def _tokenize(text: str) -> List[str]:
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z0-9\s\-.,;:!?]", " ", text)
    text = re.sub(r"([.,;:!?])", r" \1 ", text)
    return [t for t in text.split() if t]


def _match_phrases(tokens: List[str], max_len: int = 3) -> List[Tuple[int, int, str]]:
    """Greedy longest-match of vocabulary phrases. Returns (start, end, label|None)."""
    out, i = [], 0
    while i < len(tokens):
        hit = None
        for n in range(max_len, 0, -1):
            if i + n > len(tokens):
                continue
            words = tokens[i:i + n]
            if any(w in ".,;:!?" for w in words):
                continue
            phrase = " ".join(words)
            sing = " ".join(words[:-1] + [singularize(words[-1])])
            for cand in (phrase, sing):
                if cand in BLOCKED_COMPOUNDS:
                    hit = (i, i + n, BLOCKED_COMPOUNDS[cand], True)
                    break
                if cand in SYNONYMS:
                    hit = (i, i + n, SYNONYMS[cand], False)
                    break
                if n == 1 and cand in OUT_OF_VOCAB:
                    hit = (i, i + n, cand, False)
                    break
            if hit:
                break
        if hit:
            out.append(hit[:3])
            i = hit[1]
        else:
            i += 1
    return out


def _negated_positions(tokens: List[str]) -> set:
    neg = set()
    for i, t in enumerate(tokens):
        if t in NEGATORS:
            # "no" used as a sign text ("NO PARKING") is still negation of that noun: fine.
            for j in range(i + 1, min(len(tokens), i + 1 + NEG_WINDOW)):
                if tokens[j] in ".,;:!?" or tokens[j] in CLAUSE_BREAKS:
                    break
                neg.add(j)
    return neg


def parse_caption(caption: str) -> Dict[str, Any]:
    tokens = _tokenize(caption)
    negated = _negated_positions(tokens)
    objects, raw_objects, neg_objects = [], [], []

    for start, end, label in _match_phrases(tokens):
        if label is None:
            continue
        surface = " ".join(tokens[start:end])
        # the colour "orange" is not the fruit unless plural or followed by nothing noun-like
        if label == "orange" and surface == "orange":
            nxt = tokens[end] if end < len(tokens) else "."
            if nxt not in ".,;:!?" and nxt not in {"slice", "slices", "wedge", "wedges"}:
                continue
        # "baby elephant", "mother giraffe", "passenger train": the person word is a modifier
        if label == "person" and end < len(tokens):
            nxt = SYNONYMS.get(tokens[end]) or SYNONYMS.get(singularize(tokens[end]))
            if nxt and nxt != "person":
                continue
        if any(p in negated for p in range(start, end)):
            neg_objects.append(label)
            continue
        objects.append(label)
        raw_objects.append(surface)

    for t in tokens:
        if t in NO_PERSON_WORDS and NO_PERSON_WORDS[t]:
            neg_objects.append(NO_PERSON_WORDS[t])

    def dedupe(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "objects": dedupe(objects),
        "raw_objects": dedupe(raw_objects),
        "negated": dedupe(neg_objects),
        "colors": sorted({"gray" if t == "grey" else t for t in tokens if t in COLORS}),
        "counts": [{"word": t, "value": COUNT_WORDS[t]} for t in tokens if t in COUNT_WORDS],
        "raw": caption,
    }


if __name__ == "__main__":
    tests = [
        "There are no people or man-made structures visible.",
        "A bedside table with a table lamp next to a bed.",
        "A bus stop with pedestrians and a red double-decker bus.",
        "A banana plant with green bananas against the sky.",
        "A surfer rides a wave while ducks float nearby.",
        "A train with several train cars passes a car.",
        "A man in an orange shirt holds two oranges and a plane flies overhead.",
        "A sign that says NO TRUCKS above the highway.",
        "A room with a couch but no television.",
    ]
    for t in tests:
        r = parse_caption(t)
        print(f"{t}\n   -> {r['objects']}   negated={r['negated']}")
