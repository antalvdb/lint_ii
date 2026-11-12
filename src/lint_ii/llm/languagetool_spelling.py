"""
LanguageTool spelling and grammar check, next to Hunspell and the LLM pass.

LanguageTool (LGPL, Java) runs as a separate LOCAL server; set
LINT_II_LANGUAGETOOL_URL (e.g. http://127.0.0.1:8081) to switch the pass on.
Unset, or the server down, the pass returns nothing and the analysis goes on
without it. Never point it at the public languagetool.org API: tester texts
can hold personal data.

Why (scripts/eval/README.md, "LanguageTool vs Hunspell", 2026-11-11): on
corpus_spelling.json LanguageTool fixed 27 of 34 errors, Hunspell 10, the
Hunspell + Qwen pipeline 22, and pipeline + LanguageTool 29. Its gain is fixed
rules for errors whose wrong form is itself a word (d/t: "Hij vind de",
"Dat gebeurd"; "Hun hebben"; "het meisje die"), which a dictionary cannot
see, and it does not depend on the LLM provider.

What is used:
- Rule categories with errors (typos, slips, agreement, form errors); casing,
  readability, style and punctuation rules are left out (they flagged
  "Maandag" opening a list line, every long word, "ter plekke").
- Only fixes that change ONE token, so the editor shows them like a Hunspell
  fix. A fix that joins words ("politie agent" -> "politieagent") would need
  a frontend change and is skipped for now (frontend deploys are on hold
  during the judges study).
- Dictionary flags (the MORFOLOGIK rule) go through the same gates as the
  Hunspell pass, because LanguageTool also flags valid words it lacks
  (banenzwemmen -> banenzwemmer, repaircafé -> reiscafé, case numbers).
- Every fix goes through the checks shared with the LLM pass
  (SuggestionEngine._spelling_fix_rejection).
"""
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid

logger = logging.getLogger(__name__)

# Categories whose rules mark errors. Not used: CASING, LEESBAARHEID (long
# words, which LiNT measures itself), OVERIG and STYLE (word choice),
# PUNCTUATION, TYPOGRAPHY, VOLUIT_SCHRIJVEN ("t/m" -> "tot en met").
USED_CATEGORIES = frozenset({"TYPOS", "VERGISSINGEN", "CONGRUENTIE", "VORMFOUTEN", "MISC"})
DICTIONARY_RULE_PREFIX = "MORFOLOGIK_RULE"
# Rules in a used category that flag correct Dutch: "duurt kort" -> "is kort"
# fired on a clean control (eval set 3, clean-2).
DISABLED_RULES = ("NL_SIMPLE_REPLACE_DUURT_KORT",)

_DEFAULT_TIMEOUT = 10.0


def _server_url() -> str | None:
    url = os.environ.get("LINT_II_LANGUAGETOOL_URL", "").strip()
    return url.rstrip("/") or None


def enabled() -> bool:
    return _server_url() is not None


def check(text: str, url: str, timeout: float) -> list[dict]:
    """POST the text to LanguageTool's /v2/check and return its matches."""
    data = urllib.parse.urlencode({"text": text, "language": "nl",
                                   "disabledRules": ",".join(DISABLED_RULES)}).encode()
    with urllib.request.urlopen(f"{url}/v2/check", data, timeout=timeout) as resp:
        return json.load(resp)["matches"]


def one_token_fix(span: str, replacement: str) -> tuple[int, str, str] | None:
    """Reduce a match to the one token it changes: (token offset in the span's
    whitespace tokens, old token, new text), or None when it changes more
    than one token or joins or drops words. LanguageTool often matches context
    with the error ("Hij vind de" -> "Hij vindt de")."""
    old, new = span.split(), replacement.split()
    lead = 0
    while lead < min(len(old), len(new)) and old[lead] == new[lead]:
        lead += 1
    trail = 0
    while (trail < min(len(old), len(new)) - lead
           and old[-1 - trail] == new[-1 - trail]):
        trail += 1
    old_core = old[lead:len(old) - trail]
    new_core = new[lead:len(new) - trail]
    if len(old_core) != 1 or not new_core:
        return None
    return lead, old_core[0], " ".join(new_core)


def _locate(analysis, starts: list[int], offset: int):
    """(sentence index, sentence analysis, char offset in that sentence) for a
    document offset, or None when it falls between sentences."""
    for sent_idx, (start, sa) in enumerate(zip(starts, analysis.sentence_analyses)):
        if start <= offset < start + len(sa.doc.text):
            return sent_idx, sa, offset - start
    return None


def _word_index(sent_analysis, char_offset: int, token_text: str):
    # Token offsets count from the start of the whole spaCy doc, of which the
    # sentence may be a span; make them relative to the sentence.
    base = sent_analysis.doc[0].idx if len(sent_analysis.doc) else 0
    for wi, wf in enumerate(sent_analysis.word_features):
        tok = wf.token
        if tok.idx - base == char_offset and tok.text == token_text:
            return wi, wf
    return None


def generate_languagetool_suggestions(analysis, engine,
                                      existing_word_indices: set | None = None) -> list:
    """Spelling/grammar suggestions from a local LanguageTool server.

    Args:
        analysis: ReadabilityAnalysis (sentences already parsed by spaCy)
        engine: the SuggestionEngine running this analysis (its shared checks,
            defined terms and form of address)
        existing_word_indices: (sentence_idx, word_idx) pairs already covered
            by the LLM or Hunspell pass; skipped to avoid duplicates
    """
    from lint_ii.llm.hunspell_spelling import (
        _bracketed_word_part, _get_dictionary, _known_compound)
    from lint_ii.llm.suggestions import Suggestion, SuggestionEngine, SuggestionType

    url = _server_url()
    if url is None:
        return []
    covered = set(existing_word_indices or ())

    # One request for the whole document; sentences joined with a space so
    # each offset maps back to one sentence.
    starts, pos = [], 0
    for sa in analysis.sentence_analyses:
        starts.append(pos)
        pos += len(sa.doc.text) + 1
    text = " ".join(sa.doc.text for sa in analysis.sentence_analyses)
    if not text.strip():
        return []
    timeout = float(os.environ.get("LINT_II_LANGUAGETOOL_TIMEOUT", _DEFAULT_TIMEOUT))
    try:
        matches = check(text, url, timeout)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        logger.warning("LANGUAGETOOL_ERROR %s: %s (pass skipped)", url, e)
        return []

    counts: dict[str, int] = {}
    for sa in analysis.sentence_analyses:
        for wf in sa.word_features:
            counts[wf.text] = counts.get(wf.text, 0) + 1

    dictionary = None
    suggestions = []
    for m in matches:
        rule = m.get("rule", {})
        rule_id = rule.get("id", "")
        category = rule.get("category", {}).get("id", "")
        if category not in USED_CATEGORIES or not m.get("replacements"):
            continue
        offset, length = m["offset"], m["length"]
        span = text[offset:offset + length]
        fix = one_token_fix(span, m["replacements"][0]["value"])
        if fix is None:
            logger.info("LanguageTool: skipping %s '%s' -> '%s': not a one-word fix",
                        rule_id, span, m["replacements"][0]["value"])
            continue
        lead, old_token, new_text = fix
        token_offset = offset
        for tok in span.split()[:lead]:
            token_offset = text.index(tok, token_offset) + len(tok)
        token_offset = text.index(old_token, token_offset)

        located = _locate(analysis, starts, token_offset)
        if located is None:
            continue
        sent_idx, sa, char_in_sent = located
        found = _word_index(sa, char_in_sent, old_token)
        if found is None:
            logger.debug("LanguageTool: '%s' is not one spaCy token, skipped", old_token)
            continue
        word_idx, wf = found
        if (sent_idx, word_idx) in covered:
            continue
        word = wf.text  # lower-cased, as in the other spelling passes
        tok = wf.token

        if rule_id.startswith(DICTIONARY_RULE_PREFIX):
            # A dictionary flag: the same gates as the Hunspell pass.
            if dictionary is None:
                dictionary = _get_dictionary()
            if (not wf.is_content_word_excl_propn or not old_token.isalpha()
                    or old_token.isupper() or len(old_token) < 3
                    or dictionary.lookup(old_token) or dictionary.lookup(word)
                    or counts.get(word, 0) > 1 or _bracketed_word_part(tok)
                    or _known_compound(word, dictionary)
                    # A typo keeps the first letter; a different word does not
                    # (omjubeld -> bejubeld, eval set 4).
                    or new_text[:1].lower() != word[:1]
                    or not SuggestionEngine._correction_plausible(word, new_text.lower(), "spelling")):
                logger.info("LanguageTool: dictionary flag '%s' -> '%s' skipped by the "
                            "Hunspell gates", old_token, new_text)
                continue
            error_category = "spelling"
            explanation = f"'{word}' lijkt een spelfout. Bedoeld: '{new_text}'?"
        else:
            error_category = "grammar"
            explanation = m.get("message") or f"Bedoeld: '{new_text}'?"

        sent_text = sa.doc.text
        suggested_text = sent_text[:char_in_sent] + new_text + sent_text[char_in_sent + len(old_token):]
        # Match the lower-cased word: the editor restores a sentence-initial capital.
        correction = new_text
        if old_token[:1].isupper() and tok.is_sent_start and correction[:1].isupper():
            correction = correction[0].lower() + correction[1:]
        if correction == word:
            continue
        rejected = engine._spelling_fix_rejection(
            sa, word_idx, word, correction, sent_text, suggested_text)
        if rejected:
            logger.info("LanguageTool: '%s' -> '%s' skipped: %s", word, correction, rejected)
            continue

        logger.info("LanguageTool: %s '%s' -> '%s' (sentence %d)",
                    rule_id, word, correction, sent_idx)
        covered.add((sent_idx, word_idx))
        suggestions.append(Suggestion(
            id=str(uuid.uuid4())[:8],
            type=SuggestionType.SPELLING,
            sentence_index=sent_idx,
            original_text=sent_text,
            suggested_text=suggested_text,
            explanation=explanation,
            word=word,
            word_index=word_idx,
            replacement_word=correction,
            model=None,
            error_category=error_category,
        ))
    return suggestions
