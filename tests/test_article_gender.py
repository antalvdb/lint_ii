"""The article gender guard (_introduces_article_mismatch): a swap or rewrite
must not put a de-word after "het" or a het-word after "de". The study item E
swap toepassing -> gebruik gave "De door u beoogde gebruik"; the other cases
are real suggestions from the eval result files."""

import pytest

from lint_ii.llm.suggestions import SuggestionEngine as E

STUDY_E = ("De door u beoogde toepassing van gerecycled polypropyleen is volgens "
           "de regeling toegestaan.")


@pytest.mark.parametrize("original, suggested, clash", [
    (STUDY_E, STUDY_E.replace("toepassing", "gebruik"), "de gebruik"),
    ("Het vruchtbare slib had zo de tijd om zich vast te zetten.",
     "Het vruchtbare modder had zo de tijd om zich vast te zetten.", "het modder"),
    ("Het omvat het herstellen van de oorspronkelijke waterhuishouding.",
     "Het omvat het herstellen van de oorspronkelijke beheer.", "de beheer"),
    ("Op de meeste maatschappelijke terreinen bestaan adviesorganen.",
     "Op de meeste maatschappelijke terrein bestaan adviesorganen.", "de terrein"),
    ("Deze regeling geldt voor iedereen.", "Dit regeling geldt voor iedereen.", "dit regeling"),
])
def test_a_swap_that_breaks_the_article_is_caught(original, suggested, clash):
    assert E._introduces_article_mismatch(original, suggested) == clash


@pytest.mark.parametrize("original, suggested", [
    (STUDY_E, STUDY_E.replace("De door u beoogde toepassing", "Het door u bedoelde gebruik")),
    ("Het vruchtbare slib had zo de tijd.", "De vruchtbare modder had zo de tijd."),
    ("De toepassing van plastic is toegestaan.", "Het gebruik van plastic is toegestaan."),
    # The parser hangs the article on the wrong noun.
    ("De gemeente wil de uitstoot de komende jaren verminderen.",
     "De gemeente wil de uitstoot de komende tien jaar verminderen."),
    ("Hij mag zich verdedigen tegen het toegepaste geweld.",
     "Hij mag zich verdedigen tegen het door de verdachte toegepaste geweld."),
])
def test_a_fitting_article_passes(original, suggested):
    assert E._introduces_article_mismatch(original, suggested) is None


def test_a_clash_already_in_the_original_is_not_the_rewrites():
    assert E._introduces_article_mismatch(
        "De gebruik van plastic is niet toegestaan.",
        "De gebruik van plastic mag niet.") is None


@pytest.mark.parametrize("original, suggested", [
    ("Het rapport zal volgende maand aan de raad worden aangeboden.",
     "De commissie biedt het volgende maand aan de raad aan."),
    ("De reden om te sparen is dat het inkomen genereert.",
     "De reden om te sparen is dat het rente oplevert."),
])
def test_rewrites_ignore_a_pronoun_read_as_an_article(original, suggested):
    assert E._introduces_article_mismatch(original, suggested, de_only=True) is None


def test_rewrites_still_catch_de_on_a_het_word():
    assert E._introduces_article_mismatch(
        STUDY_E, "U mag de beoogde gebruik van polypropyleen toepassen.",
        de_only=True) == "de gebruik"
