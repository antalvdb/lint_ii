import re
import threading

import spacy
from spacy.language import Language

from lint_ii import LiNT_II_Exception


try:
    print('Loading Dutch language model from spaCy... ', end='')
    _RAW_NLP_MODEL : Language = spacy.load('nl_core_news_lg')
    print('✓ nl_core_news_lg')
except OSError:
    raise LiNT_II_Exception('LiNT-II requires the spaCy model "nl_core_news_lg"; download the model by running: `python -m spacy download nl_core_news_lg`')


class _ThreadSafeNLP:
    """Serialize calls into the shared spaCy pipeline.

    spaCy pipelines are not safe for concurrent calls from multiple threads,
    and the demo API runs several analyses in parallel (each does a document
    pass up front plus a short pass per generated suggestion). Pipeline calls
    take milliseconds to a few seconds, so serializing them costs little; the
    slow LLM phases stay fully parallel. Attribute access passes through, so
    tokenizer tweaks and pipeline introspection keep working."""

    def __init__(self, model: Language):
        self._model = model
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


NLP_MODEL = _ThreadSafeNLP(_RAW_NLP_MODEL)


# Keep a word with a parenthesised (letter) suffix as ONE token — e.g.
# "testuitslag(en)", "kind(eren)", "auto('s)". spaCy otherwise splits these into
# word + "(" + suffix + ")", which separated (and previously duplicated) the
# brackets in the visualiser. token_match is checked before affix stripping, so
# a trailing comma/period is still split off correctly, and a standalone
# "(parenthetical)" with spaces is unaffected. The suffix may begin with a
# straight or curly apostrophe (Dutch plurals such as auto's).
_WORD_PAREN_SUFFIX = re.compile(r"^[^\W\d_]+\(['’]?[^\W\d_]+\)$", re.UNICODE)
_prev_token_match = NLP_MODEL.tokenizer.token_match


def _token_match(text: str):
    if _WORD_PAREN_SUFFIX.match(text):
        return True
    return _prev_token_match(text) if _prev_token_match else None


NLP_MODEL.tokenizer.token_match = _token_match
