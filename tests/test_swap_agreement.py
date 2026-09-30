"""The swap agreement guard (_swap_agreement_failure): a word swap must fit
the grammatical slot of the word it replaces. Cases from the Kleijn
re-judge (README item 8); each shape is tested in both directions."""

import pytest

from lint_ii import ReadabilityAnalysis
from lint_ii.llm.suggestions import SuggestionEngine as E


def _trigger(sentence, word):
    sa = ReadabilityAnalysis.from_markdown(sentence).sentence_analyses[0]
    for t in E()._check_word_frequency(sa, 0, sa.doc.text):
        if t.word == word:
            return t
    # Not rare enough to trigger: build one from the parse directly.
    from lint_ii.llm.suggestions import SuggestionTrigger, SuggestionType
    tok = next(t for t in sa.doc if t.text == word)
    return SuggestionTrigger(
        type=SuggestionType.WORD_FREQUENCY, sentence_index=0,
        sentence_text=sa.doc.text, feature_value=1.0, threshold=3.0, word=word,
        word_index=0, word_tag=tok.tag_,
        word_left=tuple(t.text for t in sa.doc[max(0, tok.i - 2):tok.i]),
        word_verbal=E._verbal_participle(tok),
        word_nominal=E._clearly_nominal(tok))


def _check(sentence, word, replacement, written=None):
    t = _trigger(sentence, word)
    return E._swap_agreement_failure(t, replacement, sentence.replace(word, written or replacement))


@pytest.mark.parametrize("sentence, word, replacement, written", [
    ("Bij een schermutseling of een enerverende achtervolging gebruikt hij geweld.",
     "enerverende", "intens", None),
    ("Nederland kent een uitgebalanceerd systeem van rechtshandhaving.",
     "uitgebalanceerd", "goed in balans", None),
    ("Het gaat om het ontwikkelen van waarnemings- en modelsystemen, die nodig zijn.",
     "modelsystemen", "modellen", None),
    ("Er wordt verschillend gedacht over en omgegaan met homoseksualiteit.",
     "omgegaan", "omgaat", None),
    ("Daarbij mag de verdachte worden beetgepakt en in bedwang worden gehouden.",
     "beetgepakt", "vastgrijpen", None),
    ("De melder reageert op een bedieningsfout, kwaadwilligheid of laswerk.",
     "kwaadwilligheid", "opzettelijk", None),
    ("Hij kocht een uitgebalanceerd systeem.", "uitgebalanceerd", "evenwichtige", None),
    ("Een blusdeken is van glasvezel, geïmpregneerde wol of ander materiaal gemaakt.",
     "geïmpregneerde", "behandeld", None),
    ("In de steden zijn miljoenen woningen gerenoveerd en gemoderniseerd.",
     "gemoderniseerd", "modern", None),
    ("Het CBR kan een aanvullende educatieve maatregel opleggen.", "educatieve", "leerzaam", None),
    ("De levering van gas- en elektriciteitsvoorziening stopt.",
     "elektriciteitsvoorziening", "levering van elektriciteit", None),
])
def test_breaks_the_slot(sentence, word, replacement, written):
    assert _check(sentence, word, replacement, written)


@pytest.mark.parametrize("sentence, word, replacement, written", [
    # VERVANGING in base form, rewrite correctly inflected: judge the rewrite.
    ("Daarbij mag de verdachte worden beetgepakt en in bedwang worden gehouden.",
     "beetgepakt", "vastpakken", "vastgepakt"),
    ("Het is een wereldomvattende organisatie.", "wereldomvattende", "wereldwijd", "wereldwijde"),
    ("De melder reageert op een bedieningsfout, kwaadwilligheid of laswerk.",
     "kwaadwilligheid", "opzettelijk", "opzettelijk gedrag"),
    ("Bij een schermutseling of een enerverende achtervolging gebruikt hij geweld.",
     "enerverende", "spannende", None),
    ("Onze geglobaliseerde samenleving verandert.", "geglobaliseerde", "wereldwijd verbonden", None),
    ("Hij heeft het probleem omzeild.", "omzeild", "vermeden", None),
    ("Hij heeft het probleem omzeild.", "omzeild", "omringd", None),
    ("Zo verschijnen voedselgewassen op de markt.", "voedselgewassen", "gewassen voor voedsel", None),
    ("Het is een uitgebalanceerd systeem.", "uitgebalanceerd", "evenwichtig", None),
    # Unjudged Kleijn firings that were wrong (2026-09-30): after "het" the
    # -e is required; predicates and mis-tagged rare words are not nouns.
    ("Het bruto binnenlands product per hoofd is hoog.", "bruto", "totale", None),
    ("Een grote munt is equivalent aan vijf kleine muntjes.", "equivalent", "gelijk", None),
    ("Wees zuinig met de energie die voorhanden is.", "voorhanden", "beschikbaar", None),
    ("Honden kunnen rabiës krijgen.", "rabiës", "hondsdolheid", None),
    # An -en participle never inflects, so its bare form does not rule out -e.
    ("En met dat verworven geld koopt hij hout.", "verworven", "verdiende", None),
    # Adjectival participles (box probe 2026-09-30): copula, and zich voelen.
    ("Hij was nogal gepikeerd over die opmerking.", "gepikeerd", "boos", None),
    ("Ze voelde zich gepikeerd.", "gepikeerd", "beledigd", "boos"),
])
def test_fits_the_slot(sentence, word, replacement, written):
    assert _check(sentence, word, replacement, written) is None


def test_swap_that_absorbs_a_neighbour_gets_no_verdict():
    s = "Waarschuwingen voor de burger en de maritieme sector."
    t = _trigger(s, "maritieme")
    assert E._swap_agreement_failure(
        t, "scheepvaart", "Waarschuwingen voor de burger en de scheepvaartsector.") is None


def test_no_tag_fails_open():
    from lint_ii.llm.suggestions import SuggestionTrigger, SuggestionType
    t = SuggestionTrigger(type=SuggestionType.WORD_FREQUENCY, sentence_index=0,
                          sentence_text="x", feature_value=1.0, threshold=3.0, word="x")
    assert E._swap_agreement_failure(t, "intens") is None


@pytest.mark.parametrize("sentence, word, verbal", [
    ("Hij is gisteren vertrokken.", "vertrokken", True),
    ("Het huis werd verkocht.", "verkocht", True),
    ("Ik heb het boek gelezen.", "gelezen", True),
    ("Er wordt verschillend gedacht over en omgegaan met homoseksualiteit.", "omgegaan", True),
    ("Hij was nogal gepikeerd over die opmerking.", "gepikeerd", False),
    ("Ze voelde zich gepikeerd.", "gepikeerd", False),
])
def test_verbal_participle(nlp, sentence, word, verbal):
    tok = next(t for t in nlp(sentence) if t.text == word)
    assert E._verbal_participle(tok) is verbal
