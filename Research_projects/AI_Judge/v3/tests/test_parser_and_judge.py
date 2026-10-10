"""Unit tests for the v3 parser and rule judge.  Run:  python -m pytest -q tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge_v3.parse_caption_v3 import parse_caption            # noqa: E402
from judge_v3.rule_judge_v3 import rule_based_judge_v3         # noqa: E402


def objs(text):
    return parse_caption(text)["objects"]


# ---------------------------------------------------------------- parser
def test_negation_is_not_a_claim():
    r = parse_caption("There are no people or man-made structures visible.")
    assert "person" not in r["objects"] and "person" in r["negated"]


def test_negation_stops_at_clause_break():
    assert "couch" in objs("A room with a couch but no television.")
    assert "tv" not in objs("A room with a couch but no television.")


def test_sign_text_negation():
    assert "truck" not in objs("A sign that says NO TRUCKS above the highway.")


def test_people_synonyms():
    for w in ["surfer", "skier", "pedestrians", "batter", "players", "a woman", "children"]:
        assert "person" in objs(f"The photo shows {w} on the field."), w


def test_animal_and_vehicle_synonyms():
    assert "bird" in objs("Ducks float on the pond.")
    assert "airplane" in objs("A plane on the runway.")
    assert "boat" in objs("A sailboat near the dock.")


def test_blocked_compounds():
    assert "bus" not in objs("People wait at a bus stop.")
    assert "potted plant" not in objs("A banana plant with bananas.")
    assert "car" not in objs("Graffiti on a train car.")
    assert "dining table" not in objs("A table lamp on the shelf.")


def test_person_word_as_modifier():
    assert "person" not in objs("A baby elephant walks.")
    assert "elephant" in objs("A baby elephant walks.")


def test_orange_colour_vs_fruit():
    assert "orange" not in objs("An orange and white cat.")
    assert "orange" in objs("A bowl of oranges.")


def test_generic_vehicle_is_not_car():
    assert "car" not in objs("Several vehicles on the road.")


def test_markdown_is_ignored():
    assert "bus" in objs("This is a **yellow school bus** parked outside.\n*   **Object:** bus")


def test_no_duplicates():
    o = objs("A dog and another dog and three dogs.")
    assert o.count("dog") == 1


# ---------------------------------------------------------------- rule judge
def test_supported_and_unsupported():
    j = rule_based_judge_v3("A man with a dog.", [{"class": "person", "conf": 0.9}])
    st = {c["claim_normalized"]: c["status"] for c in j["claim_results"]}
    assert st == {"person": "supported", "dog": "unsupported_by_verifier"}
    assert j["hallucination_score"] == 0.5


def test_unverifiable_caption_scores_none():
    j = rule_based_judge_v3("Blue sky over a green field.", [])
    assert j["checked_claim_count"] == 0
    assert j["hallucination_score"] is None and j["grounding_score"] is None


def test_min_conf_threshold():
    dets = [{"class": "laptop", "conf": 0.2}]
    assert rule_based_judge_v3("A laptop.", dets, min_conf=0.25)["unsupported_claim_count"] == 1
    assert rule_based_judge_v3("A laptop.", dets, min_conf=0.10)["supported_claim_count"] == 1
