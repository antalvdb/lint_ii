"""Shared fixtures for the deterministic-guard test suite.

These tests cover the pure guard/filter methods on SuggestionEngine — the
deterministic backstops that reject bad LLM output. No test here makes an
LLM call; everything runs locally (spaCy, Hunspell, SUBTLEX).
"""

import os
import sys

import pytest

# Allow running from a checkout where lint_ii is not installed: the project
# uses a src layout.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope="session")
def nlp():
    """The project's shared spaCy pipeline (tokenizer tweaks included)."""
    from lint_ii.linguistic_data.nlp_model import NLP_MODEL

    return NLP_MODEL


@pytest.fixture(scope="session")
def engine():
    """A SuggestionEngine with no provider — fine for the deterministic paths."""
    from lint_ii.llm.suggestions import SuggestionEngine

    return SuggestionEngine()
