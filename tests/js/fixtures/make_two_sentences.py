"""Regenerate two_sentences.json, the editor fixture for tests/js/.

Real backend output (ReadabilityAnalysis.as_dict() plus suggestions scored by
_analyze_suggested_text), so the editor is tested on the shape it actually
receives. Rerun when that shape changes:

    python tests/js/fixtures/make_two_sentences.py

Suggestions: r0 a two-variant rewrite of sentence 0 (primary = full), r1 a
single passive rewrite of sentence 1, w1 a word-level swap, c1 a connective
merging both sentences, with composed_metrics for r0's primary text.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from lint_ii import ReadabilityAnalysis  # noqa: E402
from lint_ii.llm.suggestions import SuggestionEngine as E  # noqa: E402

S0 = "De fiets ging vorige week onverwacht kapot door een defect aan de ketting."
S1 = "Hij kwam daardoor veel te laat op zijn werk."
CONS = "De fiets ging vorige week kapot door een defect aan de ketting."
FULL = "De fiets ging vorige week kapot. De ketting was defect."
MERGE = S0[:-1] + ", dus hij kwam veel te laat op zijn werk."
RW1 = "Daardoor kwam hij veel te laat op zijn werk."
W1 = S1.replace("werk", "baan")

m = E._analyze_suggested_text
data = ReadabilityAnalysis.from_text(S0 + " " + S1).as_dict()
data["suggestions"] = {
    "suggestions": [
        {"id": "r0", "type": "sentence_rewrite", "sentence_index": 0,
         "original_text": S0, "suggested_text": FULL, "new_sentence_metrics": m(FULL),
         "explanation": "",
         "variants": [
             {"key": "conservative", "label": "Behoudend", "suggested_text": CONS,
              "new_sentence_metrics": m(CONS)},
             {"key": "full", "label": "Volledig", "suggested_text": FULL,
              "new_sentence_metrics": m(FULL)},
         ]},
        {"id": "r1", "type": "passive", "sentence_index": 1, "original_text": S1,
         "suggested_text": RW1, "new_sentence_metrics": m(RW1), "explanation": ""},
        {"id": "w1", "type": "word_frequency", "sentence_index": 1, "original_text": S1,
         "suggested_text": W1, "new_sentence_metrics": m(W1), "explanation": ""},
        {"id": "c1", "type": "connective", "sentence_index": 0, "merges_sentences": [0, 1],
         "original_text": S0 + " " + S1, "suggested_text": MERGE, "relation": "gevolg",
         "new_sentence_metrics": m(MERGE),
         "composed_metrics": {"r0": m(E._compose_merge_text(S0 + " " + S1, MERGE, FULL))}},
    ],
    "triggers_found": 4, "triggers_processed": 4, "model": "fixture",
}

with open(os.path.join(HERE, "two_sentences.json"), "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
