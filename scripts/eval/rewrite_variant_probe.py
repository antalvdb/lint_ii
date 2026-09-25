#!/usr/bin/env python3
"""Probe for consolidated-rewrite variants: does asking for a third variant
(TUSSENVORM, exactly two sentences) degrade the existing two?

Built for the feat/intermediate-split branch, but ref-agnostic: it compares the
`sentence_rewrite` prompt at two git refs, holding EVERYTHING else fixed.

    python3 rewrite_variant_probe.py --old main --new origin/feat/intermediate-split \\
        --reps 6 --workers 2

FIDELITY. Test sentences are the ones production would actually send to this
prompt: the corpus text is analysed, `identify_triggers` + `_plan_jobs` are
run, and only `consolidated` jobs (a sentence with >= 2 sentence-level issues)
are kept. The issue list comes from the engine's own `_format_issue`, the
system prompt gets the same `_append_level_constraint` for the text's LiNT
level, and responses go through the real `parse_llm_response`. Only the prompt
template differs between the two arms. The backstop, issue formatting and
engine are taken from the working tree for both, which is valid only while the
branch leaves them unchanged — verify that before trusting a comparison.

WHAT IT MEASURES, per variant, as phenomenon counts (presence/absence cannot
see this feature at all):
  present      the field came back non-empty
  sentences    sentence count of the text (the engine's own parser)
  backstop     passes `_rewrite_backstop_failure` (no-op, broken conjunction,
               altered URL, introduced misspelling, de/het) — the engine's gate
  words        word count relative to the original: a drop means content lost

and for the new arm, TUSSENVORM usability: present, exactly two sentences,
and distinct from BOTH other variants — the three conditions under which the
pipeline would actually offer it.

Endpoint via _probe_endpoint (LINT_II_PROBE_*), default Hetzner/Qwen.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
import _probe_endpoint as _ep  # noqa: E402

VARIANT_FIELDS = ("BEHOUDEND", "TUSSENVORM", "VOLLEDIG")


def template_at(ref: str) -> dict:
    """The sentence_rewrite PromptTemplate as it exists at a git ref."""
    if ref.startswith("file:"):
        # a candidate prompts.py not yet committed anywhere (prompt iteration)
        src = open(ref[5:], encoding="utf-8").read()
    else:
        src = subprocess.run(
            ["git", "show", f"{ref}:src/lint_ii/llm/prompts.py"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    ns: dict = {}
    exec(compile(src, f"prompts@{ref}", "exec"), ns)
    return ns["PROMPT_TEMPLATES"]["sentence_rewrite"], ns["parse_llm_response"]


def build_cases(corpora, engine, limit):
    """(case_id, sentence, issues, level) for every consolidated job."""
    from lint_ii import ReadabilityAnalysis

    cases = []
    for name in corpora:
        tag = os.path.basename(name).replace(".json", "").replace("corpus", "c")
        for item in json.load(open(os.path.join(HERE, name), encoding="utf-8"))["items"]:
            if not item["id"].startswith(("long-", "multi-")):
                continue
            analysis = ReadabilityAnalysis.from_text(item["text"])
            level = getattr(analysis.lint, "level", None)
            jobs = engine._plan_jobs(engine.identify_triggers(analysis), None, True)
            for job in jobs:
                if job.kind != "consolidated":
                    continue
                lines = [ln for t in job.triggers if (ln := engine._format_issue(t))]
                if not lines:
                    continue
                cases.append((f"{tag}-{item['id']}", job.triggers[0].sentence_text,
                              "\n".join(f"- {ln}" for ln in lines), level))
    return cases[:limit] if limit else cases


def n_sentences(text: str) -> int | None:
    if not text:
        return None
    from lint_ii import ReadabilityAnalysis
    try:
        return len(ReadabilityAnalysis.from_text(text).sentence_analyses)
    except Exception:  # noqa: BLE001
        return None


def norm(t: str) -> str:
    return " ".join(re.sub(r"[^\w ]", " ", (t or "").lower()).split())


def call(system: str, user: str):
    import httpx
    r = httpx.post(_ep.base_url() + "/chat/completions", headers=_ep.headers(),
                   json=_ep.body([{"role": "system", "content": system},
                                  {"role": "user", "content": user}], 1500),
                   timeout=180.0)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"] or "", d.get("usage") or {}


def probe(engine, arm, tpl, parse, case):
    cid, sentence, issues, level = case
    system = engine._append_level_constraint(tpl["system"], level)
    user = tpl["user"].format(sentence=sentence, issues=issues)
    try:
        content, usage = call(system, user)
    except Exception as e:  # noqa: BLE001
        return {"arm": arm, "cid": cid, "error": f"{type(e).__name__}: {e}"}
    parsed = parse(content, "sentence_rewrite")
    out = {"arm": arm, "cid": cid, "error": None, "usage": usage,
           "orig_words": len(sentence.split())}
    for f in VARIANT_FIELDS:
        text = engine._clean_variant(sentence, parsed.get(f, "")) if parsed.get(f) else ""
        out[f] = {
            "text": text,
            "present": bool(text),
            "sentences": n_sentences(text),
            "backstop": (engine._rewrite_backstop_failure(sentence, text) if text else "absent"),
            "words": len(text.split()) if text else 0,
        }
    return out


def summarise(rows, arm):
    rows = [r for r in rows if r["arm"] == arm and not r["error"]]
    n = len(rows)
    print(f"\n=== arm: {arm}   ({n} scored responses)")
    if not n:
        return
    for f in VARIANT_FIELDS:
        vs = [r[f] for r in rows]
        present = sum(v["present"] for v in vs)
        if not present:
            print(f"  {f:10s} not produced by this prompt")
            continue
        sents = [v["sentences"] for v in vs if v["present"] and v["sentences"]]
        passed = sum(1 for v in vs if v["present"] and v["backstop"] is None)
        ratio = [v["words"] / r["orig_words"] for v, r in zip(vs, rows) if v["present"]]
        dist = collections.Counter(sents)
        print(f"  {f:10s} present {present}/{n}   backstop-pass {passed}/{present}   "
              f"words×{statistics.mean(ratio):.2f}   sentences "
              + " ".join(f"{k}:{dist[k]}" for k in sorted(dist)))
    fails = collections.Counter(
        (f, r[f]["backstop"]) for r in rows for f in VARIANT_FIELDS
        if r[f]["present"] and r[f]["backstop"] not in (None, "absent"))
    for (f, why), c in fails.most_common():
        print(f"      backstop fail: {f} × {c}  — {why}")
    one = sum(1 for r in rows if r["BEHOUDEND"]["present"] and r["BEHOUDEND"]["sentences"] == 1)
    print(f"  BEHOUDEND honours 'one sentence': {one}/{sum(r['BEHOUDEND']['present'] for r in rows)}")
    if any(r["TUSSENVORM"]["present"] for r in rows):
        usable = sum(
            1 for r in rows
            if r["TUSSENVORM"]["present"] and r["TUSSENVORM"]["sentences"] == 2
            and norm(r["TUSSENVORM"]["text"]) not in (norm(r["BEHOUDEND"]["text"]),
                                                     norm(r["VOLLEDIG"]["text"]))
        )
        dup = sum(1 for r in rows if r["TUSSENVORM"]["present"]
                  and norm(r["TUSSENVORM"]["text"]) == norm(r["VOLLEDIG"]["text"]))
        print(f"  TUSSENVORM usable (2 sentences AND distinct): {usable}/{n}"
              f"   duplicates VOLLEDIG: {dup}/{n}")
    pt = [r["usage"].get("prompt_tokens", 0) for r in rows]
    ct = [r["usage"].get("completion_tokens", 0) for r in rows]
    print(f"  tokens/call: prompt {statistics.mean(pt):.0f}  completion {statistics.mean(ct):.0f}"
          f"  total {statistics.mean(pt) + statistics.mean(ct):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="main")
    ap.add_argument("--new", required=True)
    ap.add_argument("--corpora", default="corpus3.json,corpus4.json,corpus5.json")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    ap.add_argument("--cases", default=None,
                    help="comma-separated case ids, e.g. c3-long-1,c5-multi-6")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dump", default=None, help="write every response as JSON here")
    args = ap.parse_args()

    from lint_ii.llm.suggestions import SuggestionEngine
    engine = SuggestionEngine()
    arms = {"old": template_at(args.old), "new": template_at(args.new)}
    cases = build_cases(args.corpora.split(","), engine, args.limit)
    if args.cases:
        want = {c.strip() for c in args.cases.split(",")}
        cases = [c for c in cases if c[0] in want]
        missing = want - {c[0] for c in cases}
        if missing:
            raise SystemExit(f"not consolidated cases: {sorted(missing)}")
    print(f"{len(cases)} consolidated-rewrite cases x {args.reps} reps x 2 arms "
          f"= {len(cases) * args.reps * 2} calls   ({_ep.describe()})")
    for cid, s, _i, lvl in cases:
        print(f"  {cid:14s} L{lvl}  {len(s.split()):2d} words  {s[:70]}")

    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(probe, engine, arm, tpl, parse, c)
                for arm, (tpl, parse) in arms.items()
                for c in cases for _ in range(args.reps)]
        for f in cf.as_completed(futs):
            rows.append(f.result())
    errs = [r for r in rows if r["error"]]
    if errs:
        print(f"\n!! {len(errs)} calls failed (excluded from the counts below):")
        for e in collections.Counter(r["error"][:70] for r in errs).most_common(3):
            print(f"   {e[1]}x {e[0]}")
    for arm in arms:
        summarise(rows, arm)
    if args.dump:
        json.dump(rows, open(args.dump, "w"), ensure_ascii=False, indent=1)
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
