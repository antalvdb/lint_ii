"""Ground-truth extraction from easy/difficult text pairs (scripts/eval/
kleijn_truth.py), on invented sentences: the Kleijn texts themselves are
private and never enter the repo."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "eval"))

from kleijn_truth import extract_pair, score  # noqa: E402


def only(changes, kind):
    return [c for c in changes if c["kind"] == kind]


class TestConnectives:
    def test_removal_that_splits_a_sentence_is_a_merge_target(self):
        mak = "Hij is bang dat hij zijn baan kwijtraakt, want er is weinig werk. Hij zoekt iets anders."
        moei = "Hij is bang dat hij zijn baan kwijtraakt. Er is weinig werk. Hij zoekt iets anders."
        (c,) = only(extract_pair(mak, moei, "connectives")["changes"], "connective_removed")
        assert (c["connective"], c["relation"], c["position"]) == ("want", "reden", "between")
        assert c["merge_at"] == 0 and c["expected"]

    def test_sentence_initial_removal_is_a_merge_target(self):
        mak = "De trein had een uur vertraging. Daardoor miste hij de laatste bus naar huis."
        moei = "De trein had een uur vertraging. Hij miste de laatste bus naar huis."
        (c,) = only(extract_pair(mak, moei, "connectives")["changes"], "connective_removed")
        assert (c["relation"], c["position"], c["merge_at"], c["expected"]) == ("gevolg", "between", 0, True)

    def test_mid_sentence_adverb_is_not_expected(self):
        mak = "De dijk is verhoogd. Het water stijgt namelijk elk jaar een beetje."
        moei = "De dijk is verhoogd. Het water stijgt elk jaar een beetje."
        (c,) = only(extract_pair(mak, moei, "connectives")["changes"], "connective_removed")
        assert c["position"] == "within" and c["merge_at"] is None and not c["expected"]

    def test_relation_outside_the_whitelist_is_not_expected(self):
        mak = "De school heeft een nieuwe gymzaal. Daarnaast komt er een grotere kantine."
        moei = "De school heeft een nieuwe gymzaal. Er komt een grotere kantine."
        (c,) = only(extract_pair(mak, moei, "connectives")["changes"], "connective_removed")
        assert c["relation"] == "toevoeging" and c["position"] == "between" and not c["expected"]


def test_word_swaps_keep_the_easy_original():
    mak = "De soldaten waren enthousiast over het plan."
    moei = "De militairen waren gretig over het plan."
    swaps = only(extract_pair(mak, moei, "word_frequency")["changes"], "word_swap")
    assert {(tuple(c["mak"]), tuple(c["moei"])) for c in swaps} == {
        (("soldaten",), ("militairen",)), (("enthousiast",), ("gretig",))}
    assert all(c["sentence"] == 0 for c in swaps)


def test_reordering_records_both_dependency_lengths():
    mak = "Er waren tweehonderd mannen nodig om het schip te laten varen."
    moei = "Er waren om het schip te laten varen tweehonderd mannen nodig."
    changes = only(extract_pair(mak, moei, "word_order")["changes"], "reordered")
    assert changes
    assert all({"max_sdl_mak", "max_sdl_moei"} <= set(c) for c in changes)


def _truth():
    return {"texts": {
        "T01": {"manipulation": "word_frequency", "n_sentences_mak": 2, "n_sentences_moei": 2, "changes": [
            {"kind": "word_swap", "sentence": 0, "mak": ["soldaten"], "moei": ["militairen"]},
            {"kind": "word_swap", "sentence": 1, "mak": ["enthousiast"], "moei": ["gretig"]}]},
        "T02": {"manipulation": "connectives", "n_sentences_mak": 2, "n_sentences_moei": 3, "changes": [
            {"kind": "connective_removed", "sentence": 1, "mak": ["want"], "moei": [], "connective": "want",
             "relation": "reden", "position": "between", "merge_at": 0, "expected": True},
            {"kind": "connective_removed", "sentence": 2, "mak": ["namelijk"], "moei": [], "connective": "namelijk",
             "relation": "reden", "position": "within", "merge_at": None, "expected": False}]},
        "T03": {"manipulation": "word_order", "n_sentences_mak": 3, "n_sentences_moei": 3, "changes": [
            {"kind": "reordered", "sentence": 0, "mak": [], "moei": [], "max_sdl_mak": 4, "max_sdl_moei": 9},
            {"kind": "reordered", "sentence": 1, "mak": [], "moei": [], "max_sdl_mak": 5, "max_sdl_moei": 5}]},
    }}


def test_scorer_counts_hits_against_the_truth():
    results = {
        "T01_moei": [
            {"type": "word_frequency", "sentence_index": 0, "word": "militairen",
             "replacement_word": "soldaten", "suggested_text": "De soldaten kwamen."},
            {"type": "sentence_rewrite", "sentence_index": 1, "suggested_text": "Ze waren blij."},
        ],
        "T01_mak": [],
        "T02_moei": [{"type": "connective", "sentence_index": 0, "relation": "gevolg"}],
        "T03_moei": [{"type": "sentence_rewrite", "sentence_index": 0},
                     {"type": "max_sdl", "sentence_index": 2}],
    }
    out = score(_truth(), results)
    wf = out["word_frequency"]
    assert (wf["hard_words"], wf["targeted"], wf["restored"], wf["changed_any"]) == (2, 1, 1, 2)
    assert out["connectives"] == {"expected": 1, "found": 1, "relation_match": 0}
    wo = out["word_order"]
    assert (wo["manipulated"], wo["fired"], wo["measurable"], wo["measurable_fired"]) == (2, 1, 1, 1)
    assert (wo["other"], wo["other_fired"]) == (1, 1)
    assert out["per_sentence"]["word_frequency/moei"] == [2, 2]
    assert out["per_sentence"]["word_frequency/mak"] == [0, 2]


def test_easy_versions_alone_still_count_per_sentence():
    """The easy and difficult versions are run separately; scoring a file that
    holds only easy versions must still report their suggestions per sentence
    (the over-editing check), with no recall counted."""
    out = score(_truth(), {"T01_mak": [{"type": "max_sdl", "sentence_index": 0}],
                           "T03_mak": []})
    assert out["per_sentence"] == {"word_frequency/mak": [1, 2], "word_order/mak": [0, 3]}
    assert out["word_frequency"]["hard_words"] == 0 and out["word_order"]["manipulated"] == 0
