"""
rule_judge_v3 — YOLO rule judge with the v3 parser and fixed aggregation.

Changes vs rule_judge_v2:
  * uses parse_caption_v3 (negation, compounds, synonyms)
  * when a caption has NO checkable claims, scores are None ("unverifiable"),
    not 0.0/0.0 — so these images are excluded from averages instead of
    silently pulling both grounding and hallucination toward 0.
  * optional min_conf lets you raise the YOLO threshold without re-running YOLO.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:                                   # used as a package: judge_v3.rule_judge_v3
    from .parse_caption_v3 import COCO_SET, parse_caption
except ImportError:                    # used with judge_v3/ on sys.path
    from parse_caption_v3 import COCO_SET, parse_caption


def rule_based_judge_v3(caption: str, detections: List[Dict[str, Any]],
                        min_conf: float = 0.0) -> Dict[str, Any]:
    claims = parse_caption(caption)
    det = {str(d.get("class", "")).lower().strip() for d in detections
           if d.get("class") and float(d.get("conf", d.get("confidence", 1.0))) >= min_conf}

    results, checked, supported = [], 0, 0
    for c in claims["objects"]:
        if c not in COCO_SET:
            results.append({"claim_normalized": c, "status": "out_of_yolo_vocabulary",
                            "in_yolo_vocabulary": False, "rule_supported": None})
            continue
        checked += 1
        ok = c in det
        supported += ok
        results.append({"claim_normalized": c,
                        "status": "supported" if ok else "unsupported_by_verifier",
                        "in_yolo_vocabulary": True, "rule_supported": ok})

    score_h: Optional[float] = (checked - supported) / checked if checked else None
    score_g: Optional[float] = supported / checked if checked else None
    return {
        "claims": claims,
        "claim_results": results,
        "detected_classes": sorted(det),
        "checked_claim_count": checked,
        "supported_claim_count": supported,
        "unsupported_claim_count": checked - supported,
        "out_of_vocabulary_claim_count": len(results) - checked,
        "hallucination_score": score_h,
        "grounding_score": score_g,
        "verifiable": checked > 0,
        "judge_version": "rule_judge_v3",
    }
