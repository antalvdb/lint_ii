"""Guards that need the spaCy parse, the Hunspell dictionary, or both.

Slower than test_text_guards.py (the session fixture loads nl_core_news_lg
once), but still orders of magnitude cheaper than the eval runs that used to
be these guards' only validation.
"""

from lint_ii.llm.suggestions import SuggestionEngine


class TestNpCoordinationList:
    def test_surface_comma_list_counts_items(self, nlp):
        doc = nlp(
            "Het museum toont oude landkaarten, zeldzame globes, "
            "historische atlassen en moderne reproducties."
        )
        result = SuggestionEngine._np_coordination_list(doc)
        assert result is not None
        n_items, span = result
        assert n_items == 4
        assert span >= 8

    def test_apposition_is_rejected(self, nlp):
        doc = nlp("De directeur, een oude bekende, opende de deur en liep weg.")
        assert SuggestionEngine._np_coordination_list(doc) is None

    def test_clause_comma_without_coordinator_is_rejected(self, nlp):
        # corpus5 conj-3 shape: the comma joins clauses, not list items.
        doc = nlp("Reserveer op tijd, want de vakantieweken lopen snel vol.")
        assert SuggestionEngine._np_coordination_list(doc) is None

    def test_plain_coordination_without_commas_is_rejected(self, nlp):
        doc = nlp("Wij verkopen brood en kaas.")
        assert SuggestionEngine._np_coordination_list(doc) is None


class TestNominalizedInfinitiveList:
    def test_het_infinitive_van_items_counted(self, nlp):
        doc = nlp(
            "Het plan omvat het dempen van oude sloten, het herstellen van "
            "kades en het wekelijks testen van de pompen."
        )
        result = SuggestionEngine._nominalized_infinitive_list(doc)
        assert result is not None
        n_items, span = result
        assert n_items == 3

    def test_single_nominalization_is_not_a_list(self, nlp):
        doc = nlp("Het dempen van sloten is duur.")
        assert SuggestionEngine._nominalized_infinitive_list(doc) is None


class TestDehetDisagreement:
    def test_e_adjective_on_indefinite_neuter_noun_is_caught(self):
        # Henk zin 5 shape: 'buitenlandse bezit' should be 'buitenlands bezit'.
        assert (
            SuggestionEngine._dehet_disagreement(
                "Wij beheren buitenlandse bezit voor klanten."
            )
            == "buitenlandse bezit"
        )

    def test_definite_determiner_licenses_the_inflection(self):
        assert (
            SuggestionEngine._dehet_disagreement(
                "Wij beheren het buitenlandse bezit voor klanten."
            )
            is None
        )

    def test_correct_sentences_pass(self):
        assert (
            SuggestionEngine._dehet_disagreement(
                "Wij verkopen houten tafels en mooie stoelen."
            )
            is None
        )


class TestIntroducesMisspelling:
    def test_garbled_new_token_is_caught(self):
        assert (
            SuggestionEngine._introduces_misspelling(
                "De fietser reed door.",
                "De fietser reed door en viel op de blorptek.",
            )
            == "blorptek"
        )

    def test_legitimate_rewrite_passes(self):
        assert (
            SuggestionEngine._introduces_misspelling(
                "De fietser reed door.", "De fietser reed verder."
            )
            is None
        )

    def test_capitalised_token_is_skipped_as_proper_noun(self):
        assert (
            SuggestionEngine._introduces_misspelling(
                "De fietser reed door.", "De fietser reed door Blorptek."
            )
            is None
        )
