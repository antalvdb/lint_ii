"""Consolidated rewrite variants: conservative / intermediate / full.

The intermediate variant is the two-sentence middle option testers asked
for. The prompt requests it; `_variant_shape_failure` enforces it. These
tests drive `_generate_consolidated_suggestion` with a canned LLM response,
so they cover the parse → backstop → dedup → shape gate → ordering path
without any provider call.
"""

import pytest

from lint_ii.llm.prompts import parse_llm_response
from lint_ii.llm.providers import LLMResponse
from lint_ii.llm.suggestions import (
    SuggestionEngine,
    SuggestionJob,
    SuggestionTrigger,
    SuggestionType,
)

ORIGINAL = (
    "De gemeente heeft besloten dat de werkzaamheden aan de Stationsstraat, "
    "die door de aannemer in opdracht van de provincie worden uitgevoerd, "
    "pas na de zomervakantie van start zullen gaan."
)
ONE = (
    "De gemeente heeft besloten dat de aannemer de werkzaamheden aan de "
    "Stationsstraat pas na de zomervakantie begint."
)
TWO = (
    "De aannemer voert de werkzaamheden aan de Stationsstraat uit in opdracht "
    "van de provincie. De gemeente heeft besloten dat hij pas na de "
    "zomervakantie begint."
)
THREE = (
    "De aannemer voert de werkzaamheden aan de Stationsstraat uit. Dat doet "
    "hij in opdracht van de provincie. De gemeente heeft besloten dat hij pas "
    "na de zomervakantie begint."
)


class _CannedProvider:
    def __init__(self, content):
        self.content = content

    def complete(self, prompt, system_prompt=None, max_tokens=None):
        return LLMResponse(content=self.content, model="canned")


def _response(behoudend=ONE, tussenvorm=TWO, volledig=THREE):
    lines = [f"BEHOUDEND: {behoudend}"]
    if tussenvorm is not None:
        lines.append(f"TUSSENVORM: {tussenvorm}")
    lines += [f"VOLLEDIG: {volledig}", "UITLEG: Zin opgesplitst en actief gemaakt."]
    return "\n".join(lines)


def _generate(engine, content):
    trigger = SuggestionTrigger(
        type=SuggestionType.SENTENCE_LENGTH,
        sentence_index=0,
        sentence_text=ORIGINAL,
        feature_value=31,
        threshold=20,
    )
    job = SuggestionJob(kind="consolidated", sentence_index=0, triggers=[trigger])
    return engine._generate_consolidated_suggestion(job, _CannedProvider(content))


def _keys(suggestion):
    return [v["key"] for v in suggestion.variants]


class TestParseTussenvorm:
    def test_three_fields_parsed(self):
        parsed = parse_llm_response(_response(), "sentence_rewrite")
        assert parsed["BEHOUDEND"] == ONE
        assert parsed["TUSSENVORM"] == TWO
        assert parsed["VOLLEDIG"] == THREE

    def test_two_field_response_still_parses(self):
        parsed = parse_llm_response(_response(tussenvorm=None), "sentence_rewrite")
        assert "TUSSENVORM" not in parsed
        assert parsed["VOLLEDIG"] == THREE


class TestVariantShapeFailure:
    @pytest.mark.parametrize("n", [1, 3])
    def test_intermediate_with_wrong_count_fails(self, n):
        assert SuggestionEngine._variant_shape_failure(
            "intermediate", {"n_sentences": n}
        )

    def test_intermediate_with_two_sentences_passes(self):
        assert (
            SuggestionEngine._variant_shape_failure("intermediate", {"n_sentences": 2})
            is None
        )

    def test_intermediate_without_metrics_fails(self):
        assert SuggestionEngine._variant_shape_failure("intermediate", None)

    @pytest.mark.parametrize("n", [2, 3])
    def test_split_conservative_fails(self, n):
        assert SuggestionEngine._variant_shape_failure(
            "conservative", {"n_sentences": n}
        )

    def test_one_sentence_conservative_passes(self):
        assert (
            SuggestionEngine._variant_shape_failure("conservative", {"n_sentences": 1})
            is None
        )

    def test_full_has_no_count_contract(self):
        assert SuggestionEngine._variant_shape_failure("full", {"n_sentences": 5}) is None
        assert SuggestionEngine._variant_shape_failure("full", None) is None


class TestConsolidatedVariants:
    def test_three_distinct_variants_ordered_least_to_most_split(self, engine):
        s = _generate(engine, _response())
        assert _keys(s) == ["conservative", "intermediate", "full"]
        assert s.suggested_text == THREE  # primary stays the full variant
        by_key = {v["key"]: v for v in s.variants}
        assert by_key["intermediate"]["suggested_text"] == TWO
        assert by_key["intermediate"]["new_sentence_metrics"]["n_sentences"] == 2

    def test_three_sentence_intermediate_is_dropped(self, engine):
        s = _generate(engine, _response(tussenvorm=THREE + " Dat is nodig."))
        assert _keys(s) == ["conservative", "full"]

    def test_one_sentence_intermediate_is_dropped(self, engine):
        one_other = ONE.replace("begint", "start")
        s = _generate(engine, _response(tussenvorm=one_other))
        assert _keys(s) == ["conservative", "full"]

    def test_intermediate_duplicating_full_keeps_the_full_key(self, engine):
        # A sentence with only one natural split point: full is already two
        # sentences, so the model repeats it as the intermediate.
        s = _generate(engine, _response(tussenvorm=TWO, volledig=TWO))
        assert _keys(s) == ["conservative", "full"]

    def test_intermediate_failing_a_backstop_is_dropped(self, engine):
        # Still exactly two sentences, so only the misspelling backstop can
        # be what drops it.
        garbled = TWO.replace("werkzaamheden", "blorptek")
        s = _generate(engine, _response(tussenvorm=garbled))
        assert _keys(s) == ["conservative", "full"]

    def test_split_conservative_is_dropped(self, engine):
        # A BEHOUDEND that split anyway would sit under "Eén zin, niet
        # gesplitst" as the frontend's default.
        split = (
            "De aannemer begint pas na de zomervakantie aan de werkzaamheden "
            "aan de Stationsstraat. Dat heeft de gemeente besloten."
        )
        s = _generate(engine, _response(behoudend=split))
        assert _keys(s) == ["intermediate", "full"]

    def test_three_sentence_conservative_near_duplicate_of_full_is_dropped(self, engine):
        # Box case c5-long-7: BEHOUDEND came back as three sentences, worded
        # just differently enough from VOLLEDIG to pass the dedup.
        near_full = THREE.replace("Dat doet hij", "Hij doet dat")
        s = _generate(engine, _response(behoudend=near_full))
        assert _keys(s) == ["intermediate", "full"]

    def test_legacy_two_variant_response_unchanged(self, engine):
        s = _generate(engine, _response(tussenvorm=None))
        assert _keys(s) == ["conservative", "full"]
        assert s.suggested_text == THREE
