#!/usr/bin/env python3
"""Ground truth from Kleijn's easy/difficult text pairs, and a scorer for a
suggestion run on them. Backlog item 8.

Each difficult version (*_moei) differs from its easy version (*_mak) by one
manipulation. Diffing the pair tells exactly what was made harder, and the
easy version is a human simplification of it:

  word_frequency  rarer words swapped in   -> did a word suggestion target them,
                                              and propose the easy original?
  word_order      dependencies lengthened  -> did a sentence rewrite fire on
                                              the manipulated sentences?
  connectives     connectives removed      -> did the connective pass propose
                                              a merge there, with that relation?

Sentences are split exactly as the tool splits Markdown input (the title is a
heading, not prose), so a change's sentence number is the `sentence_index` a
suggestion on that sentence carries. A suggestion run must therefore send the
texts with format="markdown".

Extraction is local. The output quotes words from the texts, so it goes to the
gitignored scripts/eval/private/ (this file holds no text):

    python scripts/eval/kleijn_truth.py extract \\
        --cloze <Clozedata2016_...csv>   # only for the manipulation codes

Scoring a run later (results: {"T01_moei": [suggestion dicts], ...}):

    python scripts/eval/kleijn_truth.py score --results <run.json>
"""

import argparse
import difflib
import json
import os
import re
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

PRIVATE = os.path.join(HERE, "private", "kleijn")
TEXTS = os.path.join(PRIVATE, "texts")
TRUTH = os.path.join(PRIVATE, "truth.json")

MANIPULATION = {1: "connectives", 2: "word_order", 3: "word_frequency"}

# Relation a removed connective expressed. The connective pass only proposes
# reden / gevolg / tegenstelling (_CONNECTIVE_STRONG_RELATIONS), so a removed
# "daarnaast" is not something the tool is expected to restore.
CONNECTIVE_RELATION = {
    **dict.fromkeys(("daardoor", "daarom", "dus", "zodoende", "hierdoor", "zodat",
                     "waardoor", "derhalve", "bijgevolg"), "gevolg"),
    **dict.fromkeys(("want", "omdat", "namelijk", "immers", "doordat", "aangezien"), "reden"),
    **dict.fromkeys(("maar", "echter", "toch", "hoewel", "desondanks", "daarentegen",
                     "niettemin", "terwijl"), "tegenstelling"),
    **dict.fromkeys(("daarnaast", "bovendien", "ook", "verder", "tevens", "eveneens",
                     "vervolgens", "daarna", "ten", "eerst", "ook"), "toevoeging"),
}
EXPECTED_RELATIONS = {"reden", "gevolg", "tegenstelling"}
SENTENCE_TYPES = {"sentence_rewrite", "max_sdl", "content_words_per_clause", "abstract_nouns",
                  "passive", "subordinate_clause", "sentence_length"}

_PUNCT = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def bare(token: str) -> str:
    return _PUNCT.sub("", token).lower()


def sentences(text: str) -> list[dict]:
    """The prose sentences of a text, as the tool sees Markdown input."""
    from lint_ii import ReadabilityAnalysis
    a = ReadabilityAnalysis.from_markdown(text)
    return [{"text": s.doc.text, "max_sdl": s.max_sdl} for s in a.sentences]


def _tokens(sents: list[dict]) -> tuple[list[str], list[int], set[int]]:
    """All tokens of a text, the sentence each belongs to, and which token
    indices open a sentence."""
    toks, sent_of, starts = [], [], set()
    for si, s in enumerate(sents):
        words = s["text"].split()
        starts.add(len(toks))
        toks += words
        sent_of += [si] * len(words)
    return toks, sent_of, starts


def extract_pair(mak_text: str, moei_text: str, manipulation: str) -> dict:
    """Ground truth for one pair: what the difficult version changed, keyed by
    the difficult version's sentence numbers.

    One word-level diff over the whole pair, not sentence by sentence: removing
    "want" can split a sentence in two, and our segmenter splits a few long
    sentences itself (0.4-0.7%), so the sentence counts need not match. Each
    change is mapped to the difficult version's sentence it falls in."""
    s_mak, s_moei = sentences(mak_text), sentences(moei_text)
    a, a_sent, _ = _tokens(s_mak)
    b, b_sent, b_starts = _tokens(s_moei)
    sm = difflib.SequenceMatcher(a=[bare(t) for t in a], b=[bare(t) for t in b], autojunk=False)
    changes = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        # For a deletion j1 == j2 is the insertion point: the token after it.
        j = b_sent[j1] if j1 < len(b) else len(s_moei) - 1
        i = a_sent[i1] if i1 < len(a) else len(s_mak) - 1
        w = {"sentence": j, "mak_sentence": i,
             "mak": [bare(t) for t in a[i1:i2] if bare(t)],
             "moei": [bare(t) for t in b[j1:j2] if bare(t)],
             "opens_sentence": j1 in b_starts and j > 0}
        changes.append(_classify(w, manipulation, s_mak, s_moei))
    return {"manipulation": manipulation, "n_sentences_mak": len(s_mak),
            "n_sentences_moei": len(s_moei), "changes": changes}


def _classify(w: dict, manipulation: str, s_mak: list[dict], s_moei: list[dict]) -> dict:
    base = {k: w[k] for k in ("sentence", "mak", "moei")}
    if manipulation == "connectives":
        removed = [t for t in w["mak"] if t in CONNECTIVE_RELATION and t not in w["moei"]]
        if not removed:
            return {**base, "kind": "rewording"}  # restructuring around a removal
        relation = CONNECTIVE_RELATION[removed[0]]
        # At a sentence start in the difficult version, the connective linked
        # this sentence to the previous one: the pass would merge j-1 and j.
        # Mid-sentence adverbs ("namelijk", "echter") it cannot restore.
        between = w["opens_sentence"]
        return {**base, "kind": "connective_removed", "connective": removed[0],
                "relation": relation, "position": "between" if between else "within",
                "merge_at": w["sentence"] - 1 if between else None,
                "expected": between and relation in EXPECTED_RELATIONS}
    if manipulation == "word_order":
        return {**base, "kind": "reordered",
                "max_sdl_mak": s_mak[w["mak_sentence"]]["max_sdl"],
                "max_sdl_moei": s_moei[w["sentence"]]["max_sdl"]}
    return {**base, "kind": "word_swap"}


def extract(cloze_csv: str, text_dir: str = TEXTS, out: str = TRUTH) -> dict:
    import pandas as pd
    codes = (pd.read_csv(cloze_csv, usecols=["Tekstnr", "Manipulatie"], encoding="utf-8-sig")
             .drop_duplicates().set_index("Tekstnr")["Manipulatie"].to_dict())
    read = lambda n: open(os.path.join(text_dir, n), encoding="utf-8-sig").read()
    texts = {}
    for nr in sorted(codes):
        t = f"T{nr:02d}"
        texts[t] = extract_pair(read(f"{t}_mak.txt"), read(f"{t}_moei.txt"), MANIPULATION[codes[nr]])
    truth = {"meta": {"generated": date.today().isoformat(), "segmentation": "from_markdown",
                      "run_format": "markdown", "texts": len(texts)},
             "texts": texts}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    return truth


def summarize(truth: dict) -> None:
    from collections import Counter
    by = {}
    for t, d in truth["texts"].items():
        by.setdefault(d["manipulation"], []).append(d)
    for m, ds in by.items():
        kinds = Counter(c["kind"] for d in ds for c in d["changes"])
        sents = sum(len({c["sentence"] for c in d["changes"]}) for d in ds)
        print(f"{m:15s} {len(ds)} texts, {sum(kinds.values())} changes {dict(kinds)}, "
              f"{sents} sentences touched")
        if m == "connectives":
            rem = [c for d in ds for c in d["changes"] if c["kind"] == "connective_removed"]
            print(f"   removed connectives: {dict(Counter(c['connective'] for c in rem).most_common(12))}")
            print(f"   relation x position: {dict(Counter((c['relation'], c['position']) for c in rem))}")
            print(f"   expected for the connective pass: {sum(c['expected'] for c in rem)} of {len(rem)}")
        if m == "word_order":
            per = {(t, c["sentence"]): c for t, d in truth["texts"].items() for c in d["changes"]
                   if c["kind"] == "reordered"}
            longer = sum(1 for c in per.values() if (c["max_sdl_moei"] or 0) > (c["max_sdl_mak"] or 0))
            print(f"   reordered sentences where our max_sdl is longer in the difficult "
                  f"version: {longer} of {len(per)}")


# ---- scoring a suggestion run -------------------------------------------------

def score(truth: dict, results: dict) -> dict:
    """Recall-style metrics of a suggestion run on the difficult versions, plus
    suggestions per sentence on both versions when the easy ones were run too."""
    out = {"word_frequency": {"hard_words": 0, "targeted": 0, "restored": 0, "changed_any": 0},
           # "measurable": reordered sentences where OUR parser finds a longer
           # max dependency in the difficult version (59% of them); the
           # sentence-structure trigger cannot be expected to see the rest.
           "word_order": {"manipulated": 0, "fired": 0, "measurable": 0, "measurable_fired": 0,
                          "other": 0, "other_fired": 0},
           "connectives": {"expected": 0, "found": 0, "relation_match": 0},
           "per_sentence": {}}
    for t, d in truth["texts"].items():
        sugg = results.get(f"{t}_moei")
        if sugg is None:
            continue
        m = d["manipulation"]
        if m == "word_frequency":
            for c in d["changes"]:
                for word in c["moei"]:
                    if word in c["mak"]:
                        continue
                    out[m]["hard_words"] += 1
                    hit = [s for s in sugg if s.get("type") == "word_frequency"
                           and s.get("sentence_index") == c["sentence"]
                           and bare(s.get("word") or "") == word]
                    out[m]["targeted"] += bool(hit)
                    out[m]["restored"] += any(bare(s.get("replacement_word") or "") in c["mak"] for s in hit)
                    out[m]["changed_any"] += any(
                        s.get("sentence_index") == c["sentence"]
                        and word not in {bare(x) for x in (s.get("suggested_text") or "").split()}
                        for s in sugg)
        elif m == "word_order":
            reordered = [c for c in d["changes"] if c["kind"] == "reordered"]
            touched = {c["sentence"] for c in reordered}
            measurable = {c["sentence"] for c in reordered
                          if (c["max_sdl_moei"] or 0) > (c["max_sdl_mak"] or 0)}
            fired = {s.get("sentence_index") for s in sugg if s.get("type") in SENTENCE_TYPES}
            out[m]["manipulated"] += len(touched)
            out[m]["fired"] += len(touched & fired)
            out[m]["measurable"] += len(measurable)
            out[m]["measurable_fired"] += len(measurable & fired)
            others = set(range(d["n_sentences_moei"])) - touched
            out[m]["other"] += len(others)
            out[m]["other_fired"] += len(others & fired)
        elif m == "connectives":
            for c in d["changes"]:
                if c.get("kind") != "connective_removed" or not c["expected"]:
                    continue
                out[m]["expected"] += 1
                merges = [s for s in sugg if s.get("type") == "connective"
                          and s.get("sentence_index") == c["merge_at"]]
                out[m]["found"] += bool(merges)
                out[m]["relation_match"] += any(s.get("relation") == c["relation"] for s in merges)
        for version in ("mak", "moei"):
            s = results.get(f"{t}_{version}")
            if s is not None:
                agg = out["per_sentence"].setdefault(f"{m}/{version}", [0, 0])
                agg[0] += len(s)
                agg[1] += d[f"n_sentences_{version}"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--cloze", required=True)
    e.add_argument("--texts", default=TEXTS)
    e.add_argument("--out", default=TRUTH)
    s = sub.add_parser("score")
    s.add_argument("--results", required=True)
    s.add_argument("--truth", default=TRUTH)
    args = ap.parse_args()
    if args.cmd == "extract":
        summarize(extract(args.cloze, args.texts, args.out))
        print(f"\nwritten: {args.out} (private)")
    else:
        with open(args.truth, encoding="utf-8") as f:
            truth = json.load(f)
        with open(args.results, encoding="utf-8") as f:
            results = json.load(f)
        print(json.dumps(score(truth, results), indent=1))


if __name__ == "__main__":
    main()
