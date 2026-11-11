#!/usr/bin/env python3
"""Score spelling checkers on corpus_spelling.json (no LLM calls by default).

Each positive item carries `error: {wrong, right}`. A checker DETECTS it when
one of its flags overlaps the wrong span, and FIXES it when that flag's
(first) correction turns the span into `right`. On a control item any flag is
a false alarm.

Checkers:
  hunspell   our guarded Hunspell pass (generate_hunspell_suggestions)
  lt         a LanguageTool server (--lt http://127.0.0.1:8081), all rules
  lt-core    the same, without the CASING/readability/style categories
  pipeline   spelling suggestions in a run_eval.py results file (--results)

    python3 scripts/eval/spelling_compare.py [--lt URL] [--results FILE] [-v]

LanguageTool is not part of the service; run one locally (Java 17+,
`java -cp languagetool-server.jar org.languagetool.server.HTTPServer
--port 8081`). Do not use the public languagetool.org API on tester text.
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LT_SKIP_CATEGORIES = {"CASING", "LEESBAARHEID", "STYLE", "REDUNDANCY", "OVERIG",
                      "VOLUIT_SCHRIJVEN", "TYPOGRAPHY", "PUNCTUATION"}


def hunspell_flags(text):
    from lint_ii import ReadabilityAnalysis
    from lint_ii.llm.hunspell_spelling import generate_hunspell_suggestions

    out = []
    for s in generate_hunspell_suggestions(ReadabilityAnalysis.from_text(text)):
        start = text.find(s.word)
        out.append((start, start + len(s.word), s.replacement_word, "hunspell"))
    return out


def lt_flags(text, url, core):
    data = urllib.parse.urlencode({"text": text, "language": "nl"}).encode()
    d = json.load(urllib.request.urlopen(f"{url}/v2/check", data, timeout=60))
    out = []
    for m in d["matches"]:
        if core and m["rule"]["category"]["id"] in LT_SKIP_CATEGORIES:
            continue
        repl = m["replacements"][0]["value"] if m["replacements"] else None
        out.append((m["offset"], m["offset"] + m["length"], repl, m["rule"]["id"]))
    return out


def pipeline_flags(text, rec):
    out = []
    for s in rec["produced"]:
        if s["type"] != "spelling" or not s.get("word"):
            continue
        start = text.lower().find(s["word"].lower())
        out.append((start, start + len(s["word"]), s.get("replacement_word"), "llm"))
    return out


def judge(item, flags):
    text = item["text"]
    if "error" not in item:
        return ("false-alarm" if flags else "quiet"), flags
    wrong, right = item["error"]["wrong"], item["error"]["right"]
    ws = text.find(wrong)
    we = ws + len(wrong)
    hits = [f for f in flags if f[0] < we and f[1] > ws]
    for s, e, repl, _ in hits:
        if repl is None:
            continue
        fixed = text[:s] + repl + text[e:]
        if fixed[ws:].lower().startswith(right.lower()) or right.lower() in repl.lower():
            return "fixed", hits
    return ("detected" if hits else "missed"), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "scripts/eval/corpus_spelling.json"))
    ap.add_argument("--lt", help="LanguageTool server base URL")
    ap.add_argument("--results", help="run_eval.py results file for the pipeline column")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    items = json.load(open(args.corpus))["items"]
    checkers = {"hunspell": hunspell_flags}
    if args.lt:
        checkers["lt"] = lambda t: lt_flags(t, args.lt, core=False)
        checkers["lt-core"] = lambda t: lt_flags(t, args.lt, core=True)
    if args.results:
        res = json.load(open(args.results))["results"]
        by_text = {it["text"]: res.get(it["id"]) for it in items}
        checkers["pipeline"] = lambda t: pipeline_flags(t, by_text[t]) if by_text[t] else []

    table = {}
    for name, fn in checkers.items():
        counts = {}
        for it in items:
            verdict, hits = judge(it, fn(it["text"]))
            counts[verdict] = counts.get(verdict, 0) + 1
            if args.verbose and verdict not in ("fixed", "quiet"):
                print(f"{name:9} {it['id']:9} {verdict:11} "
                      f"{[(it['text'][s:e], r, rule) for s, e, r, rule in hits]}")
        table[name] = counts

    pos = sum("error" in it for it in items)
    neg = len(items) - pos
    print(f"\n{pos} positives, {neg} controls")
    print(f"{'checker':10} {'fixed':>6} {'detected':>9} {'missed':>7} {'false-alarm':>12}")
    for name, c in table.items():
        print(f"{name:10} {c.get('fixed', 0):>6} {c.get('detected', 0):>9} "
              f"{c.get('missed', 0):>7} {c.get('false-alarm', 0):>12}")


if __name__ == "__main__":
    main()
