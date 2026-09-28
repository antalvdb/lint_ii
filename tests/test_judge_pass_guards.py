"""Guards added after the Kleijn judge passes (README item 8): defined terms,
verbatim quotations, dt-inversion in the spelling pass, and the Hunspell
skips for clean text. Each guard is tested in both directions."""

import pytest

from lint_ii import ReadabilityAnalysis
from lint_ii.llm import hunspell_spelling as hs
from lint_ii.llm.suggestions import SuggestionEngine as E


class TestDefinedTerms:
    @pytest.mark.parametrize("sentence, expected", [
        ("Die suiker in het bloed noemen we glucose.", {"glucose"}),
        ("Migreren binnen een regio noem je intraregionale migratie (intra = binnen).",
         {"intra", "intraregionale", "migratie"}),
        ("In het zogenaamde Programma van Eisen staan de eisen.", {"programma", "eisen"}),
        ("Dit zijn de zogenoemde centsprenten.", {"centsprenten"}),
        ("Is het zicht minder dan 200 m, dan spreken we van 'dichte mist'.", {"dichte", "mist"}),
        ("Een arbeidsovereenkomst is een afspraak tussen werkgever en werknemer.",
         {"arbeidsovereenkomst"}),
    ])
    def test_definitions_are_found(self, sentence, expected):
        assert expected <= E._defined_terms([sentence])

    @pytest.mark.parametrize("sentence", [
        "Daarom is het belangrijk om te sparen.",
        "In andere gevallen is het een gebied binnen een land.",
        "Tegenwoordig is een dagje naar het strand niets bijzonders meer.",
        "Het beste is een deken van 180 x 180 cm.",
        "De beste plaats ervoor is de gang.",
        "Gelukkig is het meestal een klein wondje.",
    ])
    def test_ordinary_sentences_define_nothing(self, sentence):
        assert E._defined_terms([sentence]) == frozenset()

    def test_rewrite_dropping_a_term_is_caught(self, engine):
        engine._protected_terms = frozenset({"glucose"})
        orig = "Die suiker noemen we glucose en die zit in het bloed."
        assert engine._drops_defined_term(orig, "Die suiker zit in het bloed.") == "glucose"
        assert engine._drops_defined_term(orig, "Die suiker heet glucose. Die zit in het bloed.") is None


class TestQuotations:
    def test_altered_quotation_is_caught(self):
        orig = 'Cals zei: "Dat is geen goede zaak voor het land."'
        assert E._alters_quotation(orig, 'Cals zei: "Dat is slecht voor het land."')

    def test_kept_quotation_passes(self):
        orig = "Cals zei: ‘Dat is geen goede zaak.’ Hij  ging weg."
        assert E._alters_quotation(orig, "Hij ging weg. Cals zei: ‘Dat is geen goede zaak.’") is None

    def test_apostrophe_words_are_not_quotes(self):
        assert E._quoted_spans("Het zijn auto's en foto's van opa's huis.") == []

    def test_single_quoted_term(self):
        assert E._quoted_spans("Dan spreken we van 'dichte mist', of niet.") == ["dichte mist"]


def _first_sentence(text):
    return ReadabilityAnalysis.from_markdown(text).sentence_analyses[0]


class TestDtInversion:
    @pytest.mark.parametrize("text, word, blocked", [
        ("Wat houd je van deze film?", "houd", True),
        ("Wanneer vind je het leuk?", "vind", True),
        ("Word jij morgen wakker?", "word", True),
        ("Vind je moeder het goed?", "vind", False),
        ("Hij vind het goed.", "vind", False),
    ])
    def test_t_before_subject_je(self, text, word, blocked):
        sa = _first_sentence(text)
        idx = next(i for i, wf in enumerate(sa.word_features) if wf.text.lower() == word)
        assert E._adds_t_before_subject_je(sa, idx, word, word + "t") is blocked


class TestHunspellSkips:
    def test_compound_of_known_words(self):
        d = hs._get_dictionary()
        assert hs._known_compound("waterzuinige", d)
        assert hs._known_compound("slanghaspels", d)
        assert not hs._known_compound("acomodatie", d)

    def test_bracketed_word_part(self, nlp):
        doc = nlp("Het (geluids)niveau stijgt.")
        tok = next(t for t in doc if t.text.startswith("geluids"))
        assert hs._bracketed_word_part(tok)
        assert not hs._bracketed_word_part(next(t for t in doc if t.text == "stijgt"))

    def test_typo_still_corrected_and_protected_word_skipped(self):
        a = ReadabilityAnalysis.from_markdown("Wij boeken een acomodatie aan zee.")
        assert any(s.replacement_word == "accommodatie"
                   for s in hs.generate_hunspell_suggestions(a, set()))
        assert not hs.generate_hunspell_suggestions(a, set(), protected=frozenset({"acomodatie"}))


class TestSteering:
    from lint_ii.llm.suggestions import SuggestionTrigger as _T, SuggestionType as _Y

    def _t(self, typ, idx):
        return self._T(type=typ, sentence_index=idx, sentence_text="x",
                       feature_value=0.0, threshold=0.0)

    def test_level1_keeps_only_word_swaps(self, monkeypatch):
        monkeypatch.delenv("LINT_II_LEVEL1_REWRITES", raising=False)
        ts = [self._t(self._Y.MAX_SDL, 0), self._t(self._Y.WORD_FREQUENCY, 0),
              self._t(self._Y.PASSIVE, 1)]
        assert [t.type for t in E._steer_triggers(ts, 1)] == [self._Y.WORD_FREQUENCY]
        assert len(E._steer_triggers(ts, 2)) == 3
        monkeypatch.setenv("LINT_II_LEVEL1_REWRITES", "1")
        assert len(E._steer_triggers(ts, 1)) == 3

    def test_abstract_nouns_only_alongside_another_rewrite(self, monkeypatch):
        monkeypatch.delenv("LINT_II_ABSTRACT_NOUNS", raising=False)
        ts = [self._t(self._Y.ABSTRACT_NOUNS, 0), self._t(self._Y.WORD_FREQUENCY, 0),
              self._t(self._Y.ABSTRACT_NOUNS, 1), self._t(self._Y.MAX_SDL, 1)]
        kept = [(t.type, t.sentence_index) for t in E._steer_triggers(ts, 3)]
        assert (self._Y.ABSTRACT_NOUNS, 0) not in kept
        assert (self._Y.ABSTRACT_NOUNS, 1) in kept and len(kept) == 3
        monkeypatch.setenv("LINT_II_ABSTRACT_NOUNS", "1")
        assert len(E._steer_triggers(ts, 3)) == 4

    def test_level1_prompt_is_not_self_contradictory(self):
        out = E._append_level_constraint("P", 1)
        assert "al LiNT-niveau 1" in out and "verlaagt" not in out
        assert "naar niveau 2" in E._append_level_constraint("P", 3)


class TestSwapGroups:
    def _s(self, sid, typ, idx, word, repl):
        from lint_ii.llm.suggestions import Suggestion
        return Suggestion(id=sid, type=typ, sentence_index=idx, original_text="",
                          suggested_text="", explanation="", word=word, word_index=0,
                          replacement_word=repl)

    def test_identical_swaps_are_linked_in_document_order(self):
        from lint_ii.llm.suggestions import SuggestionType as Y
        sugs = [self._s("c", Y.WORD_FREQUENCY, 5, "Nomaden", "zwervers"),
                self._s("a", Y.WORD_FREQUENCY, 1, "nomaden", "Zwervers"),
                self._s("b", Y.WORD_FREQUENCY, 3, "nomaden", "reizigers"),
                self._s("d", Y.SPELLING, 4, "acomodatie", "accommodatie"),
                self._s("e", Y.SPELLING, 6, "acomodatie", "accommodatie")]
        E._group_identical_swaps(sugs)
        by = {s.id: s.group_ids for s in sugs}
        assert by["a"] == by["c"] == ["a", "c"]
        assert by["b"] == []
        assert by["d"] == by["e"] == ["d", "e"]
        assert sugs[0].as_dict()["group_ids"] == ["a", "c"]
        assert "group_ids" not in sugs[2].as_dict()
