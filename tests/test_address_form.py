"""The document-wide form-of-address check.

A text that addresses its reader with "u" must not receive a rewrite that
introduces "je" (or the reverse). The tool rewrites only some sentences, so a
switch in one of them would leave the text mixing both.
"""

import pytest

from lint_ii.llm.providers import LLMResponse
from lint_ii.llm.suggestions import (
    SuggestionEngine,
    SuggestionJob,
    SuggestionTrigger,
    SuggestionType,
)

U_TEXT = "U ontvangt deze brief omdat uw aanvraag is goedgekeurd. Wij nemen contact met u op."
JE_TEXT = "Je ontvangt deze brief omdat je aanvraag is goedgekeurd. We nemen contact met jou op."


class TestAddressForm:
    @pytest.mark.parametrize("text,expected", [
        (U_TEXT, "u"),
        (JE_TEXT, "je"),
        ("De gemeente heeft de aanvraag goedgekeurd.", None),
        ("U kunt bellen, of je mailt ons.", None),     # already mixed
        ("Jullie horen binnenkort van ons.", "je"),    # informal plural
    ])
    def test_detects_how_the_text_addresses_its_reader(self, text, expected):
        assert SuggestionEngine._address_form(text) == expected


@pytest.fixture
def u_engine():
    engine = SuggestionEngine()
    engine._address = "u"
    return engine


class TestIntroducesOtherAddress:
    ORIGINAL = "Uw aanvraag wordt binnen vier weken door de gemeente beoordeeld."

    def test_switching_to_je_is_caught(self, u_engine):
        assert u_engine._introduces_other_address(
            self.ORIGINAL, "De gemeente beoordeelt je aanvraag binnen vier weken."
        ) == "je"

    def test_keeping_u_passes(self, u_engine):
        assert u_engine._introduces_other_address(
            self.ORIGINAL, "De gemeente beoordeelt uw aanvraag binnen vier weken."
        ) is None

    def test_the_writers_own_je_is_not_ours_to_reject(self, u_engine):
        original = "Uw buurman zei: je hoort het binnenkort."
        assert u_engine._introduces_other_address(
            original, "Uw buurman zei dat je het binnenkort hoort."
        ) is None

    def test_no_single_form_means_no_check(self):
        engine = SuggestionEngine()
        engine._address = None
        assert engine._introduces_other_address(
            self.ORIGINAL, "De gemeente beoordeelt je aanvraag."
        ) is None

    def test_je_text_rejects_u(self):
        engine = SuggestionEngine()
        engine._address = "je"
        assert engine._introduces_other_address(
            "Je aanvraag wordt door ons beoordeeld.", "Wij beoordelen uw aanvraag."
        ) == "uw"

    def test_backstop_reports_it(self, u_engine):
        reason = u_engine._rewrite_backstop_failure(
            self.ORIGINAL, "De gemeente beoordeelt je aanvraag binnen vier weken."
        )
        assert reason and "'je'" in reason


class _Canned:
    def __init__(self, content):
        self.content = content

    def complete(self, prompt, system_prompt=None, max_tokens=None):
        return LLMResponse(content=self.content, model="canned")


def test_a_je_variant_is_dropped_from_a_u_text(u_engine):
    original = (
        "Uw aanvraag voor een parkeervergunning wordt, zodra alle stukken "
        "binnen zijn, binnen vier weken door de gemeente beoordeeld."
    )
    keeps_u = "De gemeente beoordeelt uw aanvraag binnen vier weken, zodra alle stukken binnen zijn."
    switches = "Zodra alle stukken binnen zijn, beoordeelt de gemeente je aanvraag. Dat duurt vier weken."
    content = f"BEHOUDEND: {keeps_u}\nVOLLEDIG: {switches}\nUITLEG: Zin herschreven."
    trigger = SuggestionTrigger(
        type=SuggestionType.SENTENCE_LENGTH, sentence_index=0,
        sentence_text=original, feature_value=22, threshold=20,
    )
    job = SuggestionJob(kind="consolidated", sentence_index=0, triggers=[trigger])
    s = u_engine._generate_consolidated_suggestion(job, _Canned(content))
    assert s is not None
    assert s.suggested_text == keeps_u
    assert s.variants == []  # only one survivor, so no choice is offered


def test_generate_suggestions_sets_the_form_before_any_pass():
    from lint_ii.core.readability_analysis import ReadabilityAnalysis

    class _Stop(Exception):
        pass

    engine = SuggestionEngine(provider=_Canned(""))
    seen = {}

    def spelling(analysis, provider):
        seen["address"] = engine._address
        raise _Stop

    engine.generate_spelling_suggestions = spelling
    with pytest.raises(_Stop):
        engine.generate_suggestions(ReadabilityAnalysis.from_text(U_TEXT))
    assert seen["address"] == "u"
