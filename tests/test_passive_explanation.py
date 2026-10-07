"""The explanation guard for inverted passive claims.

Qwen sometimes explains a rewrite that REMOVED a passive as "Passief gemaakt"
(Jenia, 2026-10, on the Joe Speedboot text). `_fix_passive_explanation`
compares the passive counts of original and rewrite and corrects the claim
only when the rewrite has fewer passives.
"""

import pytest

from lint_ii.llm.suggestions import SuggestionEngine

SPEEDBOOT = (
    "De vermeende terugkeer van tante Rosie naar Reetveerdegem werd als een "
    "aangename schok ervaren in de levens van onze volstrekt nutteloze mannen, "
    "waarvan ik er op dat ogenblik een in wording was."
)
SPEEDBOOT_REWRITE = (
    "De terugkeer van tante Rosie naar Reetveerdegem was een aangename "
    "verrassing voor onze nutteloze mannen, waaronder ik op dat moment ook hoorde."
)
PASSIVE = "Uw aanvraag wordt binnen vier weken door de gemeente beoordeeld."
ACTIVE = "De gemeente beoordeelt uw aanvraag binnen vier weken."

fix = SuggestionEngine._fix_passive_explanation


def test_the_reported_case_is_corrected():
    assert fix(
        SPEEDBOOT, SPEEDBOOT_REWRITE,
        "Passief gemaakt, woordkeuze vereenvoudigd, zinsstructuur verhelderd.",
    ) == "Actief gemaakt, woordkeuze vereenvoudigd, zinsstructuur verhelderd."


@pytest.mark.parametrize("explanation,expected", [
    ("Zin passief gemaakt.", "Zin actief gemaakt."),
    ("Passieve zin gemaakt en ingekort.", "Actief gemaakt en ingekort."),
    ("Omgezet naar passief.", "Omgezet naar actief."),
    ("Omgezet naar de passieve vorm.", "Omgezet naar actief."),
])
def test_inverted_wordings_are_corrected(explanation, expected):
    assert fix(PASSIVE, ACTIVE, explanation) == expected


@pytest.mark.parametrize("explanation", [
    "Passief actief gemaakt.",
    "Van passief naar actief.",
    "Passieve zin omgezet naar actief.",
    "Woordkeuze vereenvoudigd.",
    "",
])
def test_correct_explanations_are_left_alone(explanation):
    assert fix(PASSIVE, ACTIVE, explanation) == explanation


def test_a_rewrite_that_really_adds_a_passive_keeps_its_explanation():
    assert fix(ACTIVE, PASSIVE, "Passief gemaakt.") == "Passief gemaakt."


def test_equal_passive_counts_leave_the_explanation_alone():
    assert fix(PASSIVE, PASSIVE.replace("vier", "zes"), "Passief gemaakt.") == "Passief gemaakt."


def test_the_second_reported_case_is_corrected():
    original = (
        "De herleidbaarheid naar het oorspronkelijk IPv4-adres wordt beperkt door "
        "de laatste twee groepjes getallen van elk IP-adres te verwijderen."
    )
    rewrite = (
        "We beperken de herleidbaarheid naar het oorspronkelijk IPv4-adres. "
        "Hiervoor verwijderen we de laatste twee groepjes getallen van elk IP-adres."
    )
    assert fix(
        original, rewrite, "Passief gemaakt en zin opgesplitst voor betere leesbaarheid.",
    ) == "Actief gemaakt en zin opgesplitst voor betere leesbaarheid."
