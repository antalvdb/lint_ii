"""
Hunspell-based Dutch spell checking, complementing the LLM spelling pass.
Hunspell is high-precision and won't hallucinate corrections.
"""
import uuid
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DICT_PATH = Path(__file__).parent.parent / "linguistic_data" / "hunspell" / "nl"


@lru_cache(maxsize=1)
def _get_dictionary():
    from spylls.hunspell import Dictionary
    logger.info("Loading Dutch Hunspell dictionary from %s", _DICT_PATH)
    return Dictionary.from_files(str(_DICT_PATH))


def generate_hunspell_suggestions(
    analysis,
    existing_word_indices: set[tuple[int, int]] | None = None,
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

    Returns:
        List of Suggestion objects.
    """
    from lint_ii.llm.suggestions import Suggestion, SuggestionType
    from lint_ii.linguistic_data.wordlists import FREQ_DATA

    if existing_word_indices is None:
        existing_word_indices = set()

    zero_count_freq = 1.359228547196266
    dictionary = _get_dictionary()
    suggestions = []

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

            hunspell_suggestions = list(dictionary.suggest(word))
            if not hunspell_suggestions:
                continue

            correction = hunspell_suggestions[0]

            # Frequency guard: if the original IS in SUBTLEX-NL (a known rare word,
            # not a typo), only keep the suggestion if the correction is more frequent.
            word_lower = word.lower()
            if word_lower in FREQ_DATA:
                correction_freq = FREQ_DATA.get(correction.lower(), zero_count_freq)
                if correction_freq <= FREQ_DATA[word_lower]:
                    logger.debug(
                        "Hunspell suggestion skipped: '%s' is a known word and "
                        "correction '%s' is not more frequent", word, correction,
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
