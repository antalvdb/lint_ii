"""The LanguageTool pass (src/lint_ii/llm/languagetool_spelling.py).

No LanguageTool server is needed: `check` is replaced by canned matches in the
shape LanguageTool 6.6 returns (taken from real responses on
corpus_spelling.json, scripts/eval/README.md 2026-11-11).
"""
import pytest

from lint_ii.llm import languagetool_spelling as lt
from lint_ii.llm.languagetool_spelling import one_token_fix

URL = "http://127.0.0.1:8081"


def match(text, span, replacement, rule, category, nth=0):
    offset = -1
    for _ in range(nth + 1):
        offset = text.index(span, offset + 1)
    return {"offset": offset, "length": len(span),
            "replacements": [{"value": replacement}],
            "rule": {"id": rule, "category": {"id": category}},
            "message": f"Misschien bedoelt u: “{replacement}”"}


@pytest.fixture
def run(engine, monkeypatch):
    """run(text, matches_fn) -> suggestions, with the server mocked."""
    from lint_ii import ReadabilityAnalysis

    monkeypatch.setenv("LINT_II_LANGUAGETOOL_URL", URL)

    def _run(text, matches_for):
        analysis = ReadabilityAnalysis.from_text(text)
        joined = " ".join(sa.doc.text for sa in analysis.sentence_analyses)
        monkeypatch.setattr(lt, "check", lambda t, url, timeout: matches_for(joined))
        return lt.generate_languagetool_suggestions(analysis, engine)
    return _run


class TestOneTokenFix:
    @pytest.mark.parametrize("span,repl,expected", [
        ("Hij vind de", "Hij vindt de", (1, "vind", "vindt")),
        ("Ik wordt", "Ik word", (1, "wordt", "word")),
        ("Hun hebben", "Zij hebben", (0, "Hun", "Zij")),
        ("is gisteren bezorgt", "is gisteren bezorgd", (2, "bezorgt", "bezorgd")),
        ("comissie", "commissie", (0, "comissie", "commissie")),
        ("teveel", "te veel", (0, "teveel", "te veel")),
    ])
    def test_reduces_to_the_changed_token(self, span, repl, expected):
        assert one_token_fix(span, repl) == expected

    @pytest.mark.parametrize("span,repl", [
        ("politie agent", "politieagent"),      # joins two words
        ("Het fiets pad", "Het fietspad"),
        ("zonne energie", "zonne-energie"),
        ("de de", "de"),                        # drops a word
    ])
    def test_joins_and_drops_are_not_one_token_fixes(self, span, repl):
        assert one_token_fix(span, repl) is None


class TestPass:
    def test_dt_fix_lands_on_the_right_word(self, run):
        text = "De nieuwe buren zijn aardig. Hij vind de tuin erg mooi."
        out = run(text, lambda j: [match(j, "Hij vind de", "Hij vindt de",
                                         "NL_SIMPLE_REPLACE_HIJ_VIND_DE", "VERGISSINGEN")])
        assert [(s.sentence_index, s.word, s.replacement_word, s.error_category)
                for s in out] == [(1, "vind", "vindt", "grammar")]
        assert out[0].suggested_text == "Hij vindt de tuin erg mooi."

    def test_sentence_initial_capital_is_lowered_like_the_word(self, run):
        # word_features lower-case every word; the editor restores the capital.
        text = "Hun hebben de prijs gewonnen. Het hele dorp was trots."
        out = run(text, lambda j: [match(j, "Hun hebben", "Zij hebben", "HUN_HEBBEN",
                                         "VORMFOUTEN")])
        assert [(s.word, s.replacement_word, s.word_index) for s in out] == [("hun", "zij", 0)]
        assert out[0].suggested_text.startswith("Zij hebben")

    def test_style_and_casing_categories_are_ignored(self, run):
        text = "Hij kwam ter plekke kijken. Het duurde lang."
        out = run(text, lambda j: [
            match(j, "ter plekke", "daar", "TER_PLEKKE", "OVERIG"),
            match(j, "Het", "het", "SOME_CASE_RULE", "CASING"),
        ])
        assert out == []

    def test_compound_join_is_skipped(self, run):
        text = "Een politie agent hielp de man oversteken. Hij bedankte haar."
        out = run(text, lambda j: [match(j, "politie agent", "politieagent",
                                         "NL_SPACE_IN_COMPOUND_POLITIE_AGENT", "MISC")])
        assert out == []

    def test_dictionary_flag_on_a_real_typo_passes(self, run):
        text = "De comissie vergadert elke maand. De verslagen liggen bij de balie."
        out = run(text, lambda j: [match(j, "comissie", "commissie",
                                         "MORFOLOGIK_RULE_NL_NL", "TYPOS")])
        assert [(s.word, s.replacement_word, s.error_category) for s in out] == [
            ("comissie", "commissie", "spelling")]

    @pytest.mark.parametrize("text,span,repl", [
        # a compound of known words the dictionary lacks
        ("We gaan elke zaterdag banenzwemmen. Het bad is dan rustig.",
         "banenzwemmen", "banenzwemmer"),
        # a rare valid word; the "fix" changes the first letter
        ("De film werd door de critici omjubeld. De zaal was vol.",
         "omjubeld", "bejubeld"),
        # a case number
        ("Uw dossiernummer is 2026-KL-0815. Vermeld dat nummer bij elke vraag.",
         "2026-KL-0815", "2026-KL"),
    ])
    def test_dictionary_flags_on_valid_words_are_dropped(self, run, text, span, repl):
        out = run(text, lambda j: [match(j, span, repl, "MORFOLOGIK_RULE_NL_NL", "TYPOS")])
        assert out == []

    def test_shared_checks_apply(self, run):
        # LanguageTool proposing the stem after "u" would break agreement; the
        # check shared with the LLM pass (_breaks_finite_agreement) drops it.
        text = "Openingstijden vindt u op onze website. Daar staan ook de tarieven."
        out = run(text, lambda j: [match(j, "vindt u", "vind u", "MADE_UP_RULE",
                                         "VERGISSINGEN")])
        assert out == []

    def test_words_other_passes_flagged_are_skipped(self, engine, monkeypatch):
        from lint_ii import ReadabilityAnalysis

        monkeypatch.setenv("LINT_II_LANGUAGETOOL_URL", URL)
        text = "De comissie vergadert elke maand. De verslagen liggen bij de balie."
        analysis = ReadabilityAnalysis.from_text(text)
        monkeypatch.setattr(lt, "check", lambda t, url, timeout: [
            match(t, "comissie", "commissie", "MORFOLOGIK_RULE_NL_NL", "TYPOS")])
        assert lt.generate_languagetool_suggestions(analysis, engine, {(0, 1)}) == []


class TestOffOrDown:
    def test_off_without_the_env_var(self, engine, monkeypatch):
        from lint_ii import ReadabilityAnalysis

        monkeypatch.delenv("LINT_II_LANGUAGETOOL_URL", raising=False)
        monkeypatch.setattr(lt, "check", lambda *a: pytest.fail("must not be called"))
        analysis = ReadabilityAnalysis.from_text("Hij vind de tuin mooi. Het is zomer.")
        assert not lt.enabled()
        assert lt.generate_languagetool_suggestions(analysis, engine) == []

    def test_server_down_returns_nothing(self, engine, monkeypatch):
        from lint_ii import ReadabilityAnalysis

        monkeypatch.setenv("LINT_II_LANGUAGETOOL_URL", "http://127.0.0.1:9")  # nothing listens
        monkeypatch.setenv("LINT_II_LANGUAGETOOL_TIMEOUT", "2")
        analysis = ReadabilityAnalysis.from_text("Hij vind de tuin mooi. Het is zomer.")
        assert lt.generate_languagetool_suggestions(analysis, engine) == []
