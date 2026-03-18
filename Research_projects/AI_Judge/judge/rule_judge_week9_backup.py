from typing import Dict, Any, List, Set
from judge.parse_caption import parse_caption
NON_DETECTABLE = {
    "tree", "trees", "park", "field", "street", "road", "kitchen", "sky", "grass"
}
STOP = {
    "photo","image","objects","object","picture","scene",
    "detail","details","including","activity","activities",
    "a","an","the","of"
}

def rule_based_judge(caption: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    claims = parse_caption(caption)

    det_classes: Set[str] = {
        d.get("class", "").lower()
        for d in detections
        if d.get("class")
    }

    violations = []
    checked = 0
    supported = 0

    for obj in claims["objects"]:
        if obj in STOP:
            continue
        if obj in NON_DETECTABLE:
            continue

        checked += 1

        if obj in det_classes:
            supported += 1
        else:
            violations.append({
                "type": "object_not_detected",
                "claim": obj
            })

    hallucination_score = (len(violations) / checked) if checked else 0.0
    grounding_score = (supported / checked) if checked else 0.0

    if checked >= 8 and len(det_classes) <= 2:
        violations.append({
            "type": "too_many_claims_for_evidence",
            "claim_count": checked,
            "detected_classes": list(det_classes),
        })

    return {
        "claims": claims,
        "detected_classes": sorted(list(det_classes)),
        "violations": violations,
        "hallucination_score": float(hallucination_score),
        "grounding_score": float(grounding_score),
    }
