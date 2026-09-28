"""
Hunspell-based Dutch spell checking, complementing the LLM spelling pass.

Hunspell can misfire on correct-but-out-of-dictionary compounds (it rejects
the word, then suggest() offers the nearest dictionary word), so its
corrections go through the same plausibility gate as the LLM spelling pass.
"""
import uuid
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DICT_PATH = Path(__file__).parent.parent / "linguistic_data" / "hunspell" / "nl"

# spylls' suggest() does combinatorial compound decomposition whose cost
# explodes on long unknown words. On real tester text (Merel Scholman's reading
# chapter) it spun for minutes on "zelfregulatievaardigheden" (26 chars) and
# "neurocognitieve" (15), stalling the whole analysis with no error — pure
# Python CPU spin, so neither the LLM watchdog nor the httpx timeout caught it,
# and the job hung pending indefinitely. Long unknown tokens are almost always
# legitimate compounds the dictionary merely lacks, so offering a "correction"
# would be a false positive anyway. Skip suggest() above this length; the
# shortest observed explosion was 15 chars, so 14 is the empirical safe ceiling.
# The LLM spelling pass still covers genuine long-word typos.
_SUGGEST_MAX_LEN = 14


@lru_cache(maxsize=1)
def _get_dictionary():
    from spylls.hunspell import Dictionary
    logger.info("Loading Dutch Hunspell dictionary from %s", _DICT_PATH)
    return Dictionary.from_files(str(_DICT_PATH))


def _bracketed_word_part(token) -> bool:
    """True for a word part in brackets glued to another word, as in
    "(geluids)overlast": not a word of its own, so not a typo ("geluids" ->
    "geluiden" was offered, Kleijn)."""
    doc, i = token.doc, token.i
    if i == 0 or i + 1 >= len(doc):
        return False
    before, after = doc[i - 1], doc[i + 1]
    if before.text != "(" or after.text != ")":
        return False
    glued_after = after.whitespace_ == "" and i + 2 < len(doc) and doc[i + 2].is_alpha
    glued_before = i >= 2 and doc[i - 2].whitespace_ == "" and doc[i - 2].is_alpha
    return glued_after or glued_before


def _known_compound(word: str, dictionary) -> bool:
    """True when an unknown word splits into two words the dictionary knows,
    optionally with a linking -s-: a productive Dutch compound, which the
    dictionary merely lacks. Hunspell "corrected" such compounds to a nearby
    dictionary word on the (professionally edited) Kleijn texts: waterzuinige
    -> waterzuiger, slanghaspels -> slanghaspel, verbindbare -> verbindbaren,
    doorgelucht -> doorgelicht. A genuine typo (acomodatie) does not split."""
    w = word.lower()
    for i in range(3, len(w) - 2):
        left, right = w[:i], w[i:]
        if not dictionary.lookup(right):
            continue
        if dictionary.lookup(left):
            return True
        if left.endswith("s") and len(left) > 3 and dictionary.lookup(left[:-1]):
            return True
    return False


def generate_hunspell_suggestions(
    analysis,
    existing_word_indices: set[tuple[int, int]] | None = None,
    protected: frozenset[str] = frozenset(),
) -> list:
    """
    Check spelling of all tokens using Hunspell nl dictionary.

    Only flags words that Hunspell considers misspelled AND for which the
    top suggestion is more frequent in SUBTLEX-NL than the original — unless
    the original is not in SUBTLEX-NL at all (a true typo), in which case we
    trust Hunspell directly.

    Args:
        analysis: ReadabilityAnalysis object (sentences already parsed by spaCy)
        existing_word_indices: (sentence_idx, word_idx) pairs already covered by
            LLM spelling suggestions — skipped to avoid duplicates
        protected: lower-cased words of terms the text defines or introduces
            (SuggestionEngine._defined_terms); never "corrected"

    Returns:
        List of Suggestion objects.
    """
    from lint_ii.llm.suggestions import Suggestion, SuggestionType, SuggestionEngine

    if existing_word_indices is None:
        existing_word_indices = set()

    dictionary = _get_dictionary()
    suggestions = []

    # A writer does not repeat the same typo: an unknown word that occurs more
    # than once is deliberate (a term, a foreign word). Kleijn: "mental map"
    # -> "metal map", in a text about mental maps.
    counts: dict[str, int] = {}
    for sa in analysis.sentence_analyses:
        for wf in sa.word_features:
            key = wf.text.lower()
            counts[key] = counts.get(key, 0) + 1

    for sent_idx, sent_analysis in enumerate(analysis.sentence_analyses):
        sent_text = sent_analysis.doc.text

        for word_idx, wf in enumerate(sent_analysis.word_features):
            if (sent_idx, word_idx) in existing_word_indices:
                continue

            word = wf.text

            # Only check content words (excludes proper nouns, function words,
            # punctuation, numbers, symbols)
            if not wf.is_content_word_excl_propn:
                continue
            if not word.isalpha() or len(word) < 3:
                continue
            # Skip all-caps (abbreviations)
            if word.isupper():
                continue

            if dictionary.lookup(word):
                continue
            if word.lower() in protected:
                logger.debug("Hunspell: skipping '%s', a term the text defines", word)
                continue
            if counts.get(word.lower(), 0) > 1:
                logger.debug("Hunspell: skipping '%s', repeated in the text", word)
                continue
            if _bracketed_word_part(wf.token):
                logger.debug("Hunspell: skipping '%s', a bracketed word part", word)
                continue
            if _known_compound(word, dictionary):
                logger.debug("Hunspell: skipping '%s', a compound of known words", word)
                continue

            # Guard against spylls' pathological blow-up on long compounds
            # (see _SUGGEST_MAX_LEN). Only reached for words unknown to the
            # dictionary, which above this length are compounds, not typos.
            if len(word) > _SUGGEST_MAX_LEN:
                logger.debug(
                    "Hunspell: skipping suggest() for long unknown word '%s' (%d chars)",
                    word, len(word),
                )
                continue

            hunspell_suggestions = list(dictionary.suggest(word))
            if not hunspell_suggestions:
                continue

            correction = hunspell_suggestions[0]

            # Plausibility gate, shared with the LLM spelling pass. "Unknown to
            # SUBTLEX" does NOT mean "a real typo": productive Dutch compounds
            # are routinely missing from both the Hunspell dictionary and
            # SUBTLEX while being perfectly correct (vogeltelling,
            # telformulieren — eval set 4), and suggest() then "corrects" them
            # to the nearest dictionary word (vogelhelling, teelformulieren).
            # A stem-changing correction must land on a strictly more frequent
            # word; splits and same-stem inflection fixes stay allowed.
            if not SuggestionEngine._correction_plausible(word, correction, "spelling"):
                logger.debug(
                    "Hunspell suggestion skipped as implausible: '%s' -> '%s'",
                    word, correction,
                )
                continue

            suggested_text = sent_text.replace(word, correction, 1)
            if suggested_text == sent_text:
                continue

            logger.info(
                "Hunspell: '%s' → '%s' (sentence %d)", word, correction, sent_idx
            )
            suggestions.append(Suggestion(
                id=str(uuid.uuid4())[:8],
                type=SuggestionType.SPELLING,
                sentence_index=sent_idx,
                original_text=sent_text,
                suggested_text=suggested_text,
                explanation=f"'{word}' lijkt een spelfout. Bedoeld: '{correction}'?",
                word=word,
                word_index=word_idx,
                replacement_word=correction,
                model=None,
                error_category="spelling",
            ))

    return suggestions
