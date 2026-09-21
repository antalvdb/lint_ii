"""Text-level deterministic guards: no spaCy parse, no Hunspell.

Every guard is tested in BOTH directions — that it fires on a breach and
stays silent on legitimate input — because "0 violations" is
indistinguishable from "the checker never runs" (eval method lesson 4).
"""

import pytest

from lint_ii.llm.suggestions import SuggestionEngine, _levenshtein


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("woord", "woord") == 0

    def test_substitution_insertion(self):
        assert _levenshtein("word", "wordt") == 1
        assert _levenshtein("vind", "vindt") == 1
        assert _levenshtein("kat", "hond") == 4


class TestIsNoopRewrite:
    def test_case_whitespace_punctuation_only_is_noop(self):
        assert SuggestionEngine._is_noop_rewrite(
            "Dit is,  een test.", "dit is een test"
        )

    def test_real_change_is_not_noop(self):
        assert not SuggestionEngine._is_noop_rewrite(
            "Dit is een test.", "Dit is een proef."
        )


class TestBreaksClauseConjunction:
    def test_dropped_maar_join_is_caught(self):
        assert (
            SuggestionEngine._breaks_clause_conjunction(
                "Hij wilde komen, maar het regende.",
                "Hij wilde komen. Het regende echter.",
            )
            == "maar"
        )

    def test_kept_join_passes(self):
        assert (
            SuggestionEngine._breaks_clause_conjunction(
                "Hij wilde komen, maar het regende.",
                "Hij wilde graag komen, maar het regende hard.",
            )
            is None
        )

    def test_no_join_in_original_passes(self):
        assert (
            SuggestionEngine._breaks_clause_conjunction(
                "Het regende hard.", "Het regende."
            )
            is None
        )


class TestAltersUrl:
    def test_dropped_url_is_caught(self):
        assert (
            SuggestionEngine._alters_url(
                "Kijk op https://voorbeeld.nl/informatie voor meer.",
                "Kijk op onze website voor meer.",
            )
            == "https://voorbeeld.nl/informatie"
        )

    def test_kept_url_passes_despite_trailing_punctuation(self):
        assert (
            SuggestionEngine._alters_url(
                "Meer op www.voorbeeld.nl.",
                "U vindt meer op www.voorbeeld.nl van de gemeente.",
            )
            is None
        )

    def test_altered_email_is_caught(self):
        assert (
            SuggestionEngine._alters_url(
                "Mail naar info@gemeente.nl met vragen.",
                "Mail naar info@gemeente.com met vragen.",
            )
            == "info@gemeente.nl"
        )


class TestFrequencyBand:
    def test_same_band_swap_rejected(self):
        # "uitstoot" 2.86 → "uitlaat" 3.38: both band 3, not a simplification.
        assert not SuggestionEngine._in_higher_freq_band(3.38, 2.86)

    def test_genuine_band_jump_accepted(self):
        assert SuggestionEngine._in_higher_freq_band(3.6, 2.4)


class TestSameWordFamily:
    def test_inflection_variants_are_family(self):
        assert SuggestionEngine._same_word_family("koelere", "koele")
        assert SuggestionEngine._same_word_family("tiental", "tientallen")

    def test_topically_related_words_are_not_family(self):
        assert not SuggestionEngine._same_word_family("koeling", "koelkast")
        assert not SuggestionEngine._same_word_family("uitstoot", "uitlaat")


class TestReplacementPassesBand:
    def test_no_replacement_passes(self, engine):
        assert engine._replacement_passes_band("woord", None, 3.0)

    def test_family_swap_rejected(self, engine):
        assert not engine._replacement_passes_band("koelere", "koele", 3.0)

    def test_band_jump_accepted(self, engine):
        # vervoerbewijs (1.84, band 2) → kaartje (4.45, band 4)
        assert engine._replacement_passes_band("vervoerbewijs", "kaartje", 1.84)

    def test_multiword_only_for_long_compounds(self, engine):
        long_compound = "levensmiddelendistributiecentrum"
        assert engine._replacement_passes_band(
            long_compound, "centrum voor levensmiddelen", 1.4
        )
        assert not engine._replacement_passes_band("besluit", "wat is besloten", 3.0)


class TestIsRecompound:
    def test_separable_verb_reglued(self):
        assert SuggestionEngine._is_recompound("kapotging", {"kapot", "ging"})

    def test_new_word_is_not_recompound(self):
        assert not SuggestionEngine._is_recompound("kapotging", {"de", "fiets"})

    def test_single_char_halves_do_not_match(self):
        assert not SuggestionEngine._is_recompound("eraan", {"e", "raan"})


class TestConnectiveAddsContent:
    ORIGINAL = "De fiets ging kapot. Hij kwam te laat op zijn werk."

    def test_added_connective_is_allowed(self):
        merged = "De fiets ging kapot, daardoor kwam hij te laat op zijn werk."
        assert SuggestionEngine._connective_adds_content(self.ORIGINAL, merged) is None

    def test_invented_content_word_is_caught(self):
        merged = "De fiets ging kapot, daardoor miste hij de vergadering."
        assert (
            SuggestionEngine._connective_adds_content(self.ORIGINAL, merged)
            == "miste"
        )

    def test_reglued_separable_verb_is_allowed(self):
        merged = "Omdat de fiets kapotging, kwam hij te laat op zijn werk."
        assert SuggestionEngine._connective_adds_content(self.ORIGINAL, merged) is None

    def test_geen_allowed_when_original_negated(self):
        original = "Hij had niet een opvolger. De zaak sloot."
        merged = "Hij had geen opvolger, dus de zaak sloot."
        assert SuggestionEngine._connective_adds_content(original, merged) is None

    def test_geen_rejected_when_original_not_negated(self):
        original = "Hij had een opvolger. De zaak sloot."
        merged = "Hij had geen opvolger, dus de zaak sloot."
        assert SuggestionEngine._connective_adds_content(original, merged) == "geen"


class TestConnectiveComposition:
    PAIR = "De fiets ging kapot. Hij kwam te laat."
    MERGE = "De fiets ging kapot, daardoor kwam hij te laat."

    def test_inserted_word_found(self):
        assert (
            SuggestionEngine._connective_inserted_word(self.PAIR, self.MERGE)
            == "daardoor"
        )

    def test_compose_merge_grafts_rewrite(self):
        composed = SuggestionEngine._compose_merge_text(
            self.PAIR, self.MERGE, "Zijn fiets ging stuk."
        )
        assert composed == "Zijn fiets ging stuk, daardoor kwam hij te laat."

    def test_compose_returns_none_without_locatable_join(self):
        assert (
            SuggestionEngine._compose_merge_text(
                self.PAIR, "De fiets ging kapot en hij kwam te laat.", "Rewrite."
            )
            is None
        )


class TestParseEnumerationResponse:
    def test_parses_intro_items_uitleg(self):
        content = (
            "INLEIDING: Het plan bestaat uit drie delen:\n"
            "ITEM: het dempen van sloten\n"
            "ITEM: - het herstellen van kades\n"
            "UITLEG: Een lijst leest makkelijker.\n"
        )
        intro, items, uitleg = SuggestionEngine._parse_enumeration_response(content)
        assert intro == "Het plan bestaat uit drie delen:"
        assert items == ["het dempen van sloten", "het herstellen van kades"]
        assert uitleg == "Een lijst leest makkelijker."

    def test_empty_items_and_separators_ignored(self):
        intro, items, uitleg = SuggestionEngine._parse_enumeration_response(
            "---\nITEM:\nonzinregel\n"
        )
        assert intro is None and items == [] and uitleg == ""


class TestEnumerationAddsContent:
    ORIGINAL = (
        "Het plan omvat het dempen van oude sloten, het herstellen van kades "
        "en het testen van de pompen."
    )

    def test_anchored_items_pass(self):
        items = ["het dempen van oude sloten", "het regelmatig testen van de pompen"]
        assert (
            SuggestionEngine._enumeration_adds_content(self.ORIGINAL, "intro", items)
            is None
        )

    def test_unanchored_item_is_returned(self):
        items = ["het dempen van oude sloten", "nieuwe speeltuinen bouwen"]
        assert (
            SuggestionEngine._enumeration_adds_content(self.ORIGINAL, "intro", items)
            == "nieuwe speeltuinen bouwen"
        )


class TestCleanVariant:
    def test_strips_list_marker_and_wrapping(self):
        assert (
            SuggestionEngine._clean_variant("De zin.", '1. "De korte zin."')
            == "De korte zin."
        )
        assert (
            SuggestionEngine._clean_variant("De zin.", "*De korte zin.*")
            == "De korte zin."
        )

    def test_keeps_quotes_the_original_had(self):
        assert (
            SuggestionEngine._clean_variant('"Kom hier."', '"Kom eens hier."')
            == '"Kom eens hier."'
        )


class TestCorrectionPlausible:
    def test_split_into_common_words_is_plausible(self):
        assert SuggestionEngine._correction_plausible("teveel", "te veel", "spelfout")

    def test_split_with_rare_part_is_rejected(self):
        # eval set 5: Hunspell proposed "terugzwemmen" → "te rugzwemmen".
        assert not SuggestionEngine._correction_plausible(
            "terugzwemmen", "te rugzwemmen", "spelfout"
        )

    def test_same_stem_inflection_is_plausible(self):
        assert SuggestionEngine._correction_plausible("word", "wordt", "grammatica")

    def test_short_function_word_swap_is_plausible(self):
        assert SuggestionEngine._correction_plausible("de", "het", "grammatica")

    def test_stem_swap_to_unknown_word_is_rejected(self):
        # eval set 4 hallucinations, arriving under a grammatica label.
        assert not SuggestionEngine._correction_plausible(
            "vogeltelling", "vogelhelling", "grammatica"
        )
        assert not SuggestionEngine._correction_plausible(
            "telformulieren", "teelformulieren", "spelfout"
        )

    def test_genuine_typo_fix_is_plausible(self):
        assert SuggestionEngine._correction_plausible(
            "acomodatie", "accommodatie", "spelfout"
        )

    def test_item6_long_unknown_compound_mutation_is_rejected(self):
        """Backlog item 6, FIXED: Hunspell turned the valid compound
        "banenzwemmen" into "banenzwemmer" -- a gerund into a person -- because
        11 of 12 characters matched. Both forms are absent from SUBTLEX, so the
        frequency rule had no signal either."""
        assert not SuggestionEngine._correction_plausible(
            "banenzwemmen", "banenzwemmer", "spelfout"
        )

    @pytest.mark.parametrize(
        "word,correction",
        [
            ("word", "wordt"),      # suffix added
            ("wordt", "word"),      # suffix removed
            ("loop", "loopt"),
            ("vind", "vindt"),
            ("aparte", "apart"),
        ],
    )
    def test_suffix_inflection_is_plausible(self, word, correction):
        # One form is a PREFIX of the other: that is what a real inflection fix
        # looks like, and it is the distinction item 6's fix turns on.
        assert SuggestionEngine._correction_plausible(word, correction, "grammatica")

    @pytest.mark.parametrize(
        "word,correction",
        [
            ("banenzwemmen", "banenzwemmer"),   # final letter substituted
            ("vogeltelling", "vogelhelling"),   # divergence mid-word
            ("telformulieren", "teelformulieren"),
        ],
    )
    def test_shared_prefix_without_prefix_relation_is_rejected(self, word, correction):
        # Sharing a long prefix and then DIVERGING is a different word, not an
        # inflection. The old test accepted these on prefix length alone.
        assert not SuggestionEngine._correction_plausible(word, correction, "spelfout")


class TestAbbreviationTokenTrigger:
    """`_check_word_frequency`'s abbreviation guard (backlog item 5, 23fbe2a).

    spaCy keeps a sentence-final word it treats as an abbreviation as ONE token
    including the period ("pas.", "vol.", "hand."). SUBTLEX has no entry for
    that form, so a perfectly common word scores ~3 Zipf too low and fires a
    spurious trigger. Observed live: clean-1 produced "pas." → "kaart", a real
    suggestion on a word that was never difficult.

    These lock in BOTH directions, because a guard that suppresses everything
    would pass a one-sided test: the abbreviation tokens must not trigger, and
    genuinely rare words must still trigger — including sentence-final ones,
    where spaCy splits the period so the token is already bare.
    """

    @staticmethod
    def _triggers(engine, text):
        from lint_ii import ReadabilityAnalysis

        analysis = ReadabilityAnalysis.from_text(text)
        found = []
        for idx, sent in enumerate(analysis.sentence_analyses):
            found += [
                t.word for t in engine._check_word_frequency(sent, idx, sent.doc.text)
            ]
        return found

    @pytest.mark.parametrize(
        "text,token",
        [
            ("U kunt boeken lenen met uw pas.", "pas."),
            ("Reserveer op tijd, want de vakantieweken lopen snel vol.", "vol."),
        ],
    )
    def test_abbreviation_token_does_not_trigger(self, engine, text, token):
        assert token not in self._triggers(engine, text)

    def test_rare_word_still_triggers(self, engine):
        assert "reprimande" in self._triggers(
            engine, "De rechter gaf de advocaat een stevige reprimande wegens de stukken."
        )

    def test_rare_word_at_sentence_end_still_triggers(self, engine):
        # spaCy splits the period here, so the token is bare and unaffected.
        assert "precair" in self._triggers(
            engine, "De financiering van het jeugdhonk blijft volgens de wethouder uiterst precair."
        )

    def test_guard_only_fires_when_the_bare_form_is_common(self):
        # The predicate itself: suppression requires the stripped form to clear
        # the threshold, so a genuinely rare abbreviation is still reported.
        from lint_ii.linguistic_data.wordlists import FREQ_DATA

        assert FREQ_DATA.get("pas", 0) >= 3.0 > FREQ_DATA.get("pas.", 0)
        assert FREQ_DATA.get("vol", 0) >= 3.0 > FREQ_DATA.get("vol.", 0)
