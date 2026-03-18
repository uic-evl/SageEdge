from __future__ import annotations
from typing import Dict, Any, List, Set

from judge.parse_caption_v2 import parse_caption
from judge.normalize_claims import (
    NON_DETECTABLE,
    normalize_detection_label,
)

STOP = {
    "photo","image","objects","object","picture","scene",
    "detail","details","including","activity","activities",
    "a","an","the","of"
}

def rule_based_judge_v2(caption: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    claims = parse_caption(caption)

    det_classes_raw: List[str] = [
        d.get("class", "").lower().strip()
        for d in detections
        if d.get("class")
    ]

    det_classes_normalized: List[str] = [
        normalize_detection_label(c)
        for c in det_classes_raw
        if c
    ]

    det_classes_set: Set[str] = set(det_classes_normalized)

    violations = []
    checked = 0
    supported = 0

    for obj in claims["objects"]:
        if obj in STOP:
            continue
        if obj in NON_DETECTABLE:
            continue

        checked += 1

        if obj in det_classes_set:
            supported += 1
        else:
            violations.append({
                "type": "object_not_detected",
                "claim_raw": obj,
                "claim_normalized": obj,
                "detected_classes_raw": sorted(set(det_classes_raw)),
                "detected_classes_normalized": sorted(det_classes_set),
            })

    hallucination_score = (len(violations) / checked) if checked else 0.0
    grounding_score = (supported / checked) if checked else 0.0

    if checked >= 8 and len(det_classes_set) <= 2:
        violations.append({
            "type": "too_many_claims_for_evidence",
            "claim_count": checked,
            "detected_classes_raw": sorted(set(det_classes_raw)),
            "detected_classes_normalized": sorted(det_classes_set),
        })

    return {
        "claims": claims,
        "detected_classes_raw": sorted(set(det_classes_raw)),
        "detected_classes": sorted(det_classes_set),
        "violations": violations,
        "hallucination_score": float(hallucination_score),
        "grounding_score": float(grounding_score),
        "checked_claim_count": checked,
        "supported_claim_count": supported,
        "unsupported_claim_count": len([v for v in violations if v["type"] == "object_not_detected"]),
    }
