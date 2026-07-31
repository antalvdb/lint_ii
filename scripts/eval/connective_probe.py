#!/usr/bin/env python3
"""Direct-to-Qwen probe for the connective prompt — prompt iteration without a
deploy round-trip.

Why this exists: a 100-item eval run cannot resolve a one-item recall change in
a 10-item phenomenon group, and a deploy cycle per prompt edit is far too slow.
This rebuilds the exact ``{paragraph}``/``{boundaries}`` pair the connective
pass sends, calls the provider with production settings, and applies the REAL
``parse_block_response`` + relation whitelist — so a hit here means the same
thing a hit in the pipeline means. One variant sweep is ~90s.

    # score the live prompt on every connective case in corpus4 + corpus5
    python3 connective_probe.py --reps 6

    # A/B a candidate against the live prompt
    python3 connective_probe.py --variant my_idea --compare base --reps 6

    # focus on a few cases
    python3 connective_probe.py --reps 8 --cases c4-conn-8,c4-conn-10

Requires HETZNER_API_KEY (box: /etc/lint-ii/lint-ii.env). Note this bypasses
the pass's own feature gate, so LINT_II_CONNECTIVES is irrelevant here.

TWO RULES THIS HARNESS EARNED (see scripts/eval/README.md):

1. Run every case 5-6x before believing a delta. At 3 reps one variant looked
   like 8/12; at 5 reps the same variant was 9/20, and a single item swung
   3/3 -> 3/5 between runs of an IDENTICAL prompt. One-shot A/Bs are noise.
2. Keep new prompt-example vocabulary clear of corpus text. An example opening
   "De zaal was tot de laatste stoel gevuld" knocked corpus5 conn-4 ("De zaal
   was ... toch uitverkocht") from 5/5 to 1/5. --audit flags such collisions.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

from lint_ii.llm.prompts import (  # noqa: E402
    PROMPT_TEMPLATES,
    parse_block_response,
)

BASE = PROMPT_TEMPLATES["connective"]

# Mirrors SuggestionGenerator._CONNECTIVE_STRONG_RELATIONS. A block whose
# RELATIE is outside this set is discarded by the pipeline, so it is not a hit.
STRONG_RELATIONS = {"reden", "gevolg", "tegenstelling"}

# Mirrors SuggestionGenerator._CONNECTIVE_LEXICON.
CONNECTIVE_LEXICON = frozenset({
    "want", "omdat", "doordat", "aangezien", "zodat", "waardoor", "dus",
    "daarom", "hierdoor", "daardoor", "maar", "echter", "toch", "hoewel",
    "terwijl", "immers", "namelijk", "bovendien", "daarnaast", "verder",
    "ook", "vervolgens", "daarna", "kortom", "derhalve", "bijgevolg",
    "desondanks", "niettemin", "integendeel", "sterker",
})


# --------------------------------------------------------------------------
# Variants
#
# "base" is always the live prompt. To try an idea, add an entry here built by
# substituting into BASE["user"]; `variant()` refuses a substitution that did
# not apply, so an edit to prompts.py can never silently turn your candidate
# back into base (this bit us: the pre-8397cae variants became no-ops the
# moment the base prompt changed).
#
# Deliberately NOT kept: the v1-v9 ladder from the 8397cae session. Those
# patched pre-8397cae prompt text and are dead against the current base. What
# they established is recorded in README.md instead — chiefly that narrowing
# the "simpele opeenvolging" clause ALONE changes nothing, because Qwen moves
# on worked examples rather than abstract guidance.
# --------------------------------------------------------------------------

def variant(old: str, new: str, *, name: str) -> str:
    """BASE["user"] with `old` replaced by `new`, or die if `old` is absent."""
    text = BASE["user"]
    if old not in text:
        raise SystemExit(
            f"variant {name!r}: anchor text not found in the current connective "
            f"prompt — prompts.py has changed under it. Re-anchor the variant "
            f"against the live prompt instead of scoring a silent no-op.\n"
            f"  anchor: {old[:90]!r}..."
        )
    return text.replace(old, new)


VARIANTS: dict[str, str] = {"base": BASE["user"]}

# Example of the intended shape (harmless: it only relabels the guard clause).
# Replace or delete freely — this is a template, not a result.
VARIANTS["example_stricter_guard"] = variant(
    "In de meeste alinea's is geen verbindingswoord nodig; antwoord dan met GEEN.",
    "In de meeste alinea's is geen verbindingswoord nodig; bij enige twijfel "
    "antwoord je met GEEN.",
    name="example_stricter_guard",
)


# --------------------------------------------------------------------------
# Cases, read from the corpora so they cannot drift out of sync
# --------------------------------------------------------------------------

# Pure temporal sequence with no causation: the shape that over-triggers first
# when the gevolg guidance is loosened. Not present in any corpus.
SYNTHETIC_NEGATIVES = [
    ("seq-1", "De wandeling begint bij de kerk. Halverwege staat een bankje bij de vijver. Het pad eindigt op de dijk."),
    ("seq-2", "De markt begint om acht uur in de ochtend. De kramen zijn om vier uur weer afgebroken."),
]


def load_cases(corpora: list[str]) -> tuple[list, list]:
    """(positives, negatives) from the given corpus files.

    Positives are the conn-* items. Negatives are every other should_suggest
    =False item that actually reaches the pass (>=2 sentences with a candidate
    boundary) — including the family-*/shortlist-* groups, which is what
    catches a new prompt example bleeding into an unrelated item.
    """
    positives, negatives = [], []
    for name in corpora:
        path = name if os.path.isabs(name) else os.path.join(HERE, name)
        tag = re.sub(r"\.json$", "", os.path.basename(path)).replace("corpus", "c")
        for item in json.load(open(path, encoding="utf-8"))["items"]:
            cid = f"{tag or 'c1'}-{item['id']}"
            if item["id"].startswith("conn-"):
                positives.append((cid, item["text"]))
            elif not item.get("should_suggest") and build(item["text"])[1]:
                negatives.append((cid, item["text"]))
    return positives, negatives


# --------------------------------------------------------------------------
# Prompt assembly — mirrors _connective_paragraphs / _connective_candidates.
# Regex sentence split rather than spaCy: the corpora are simple declarative
# prose, and this keeps the probe dependency-free and fast. If a case ever
# disagrees with the pipeline, check this first.
# --------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def build(text: str) -> tuple[str, str]:
    """Return (numbered paragraph, comma-joined 1-based candidate boundaries)."""
    sents = [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))

    def long_declarative(s: str) -> bool:
        return not s.endswith("?") and len(re.findall(r"\w+", s)) >= 4

    def opens_with_connective(s: str) -> bool:
        words = re.findall(r"\w+", s)
        return bool(words) and words[0].lower() in CONNECTIVE_LEXICON

    bounds = [
        p + 1
        for p in range(len(sents) - 1)
        if long_declarative(sents[p])
        and long_declarative(sents[p + 1])
        and not opens_with_connective(sents[p + 1])
    ]
    return numbered, ", ".join(str(b) for b in bounds)


def call_model(user_prompt: str) -> str:
    """One completion with the live provider settings (see HetznerProvider)."""
    import httpx

    key = os.environ.get("HETZNER_API_KEY")
    if not key:
        raise SystemExit("HETZNER_API_KEY not set (box: /etc/lint-ii/lint-ii.env)")
    body = {
        "model": os.environ.get("LINT_II_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8"),
        "messages": [
            {"role": "system", "content": BASE["system"]},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(os.environ.get("LINT_II_HETZNER_TEMPERATURE", "0.3")),
        "max_tokens": 2000,
        # Thinking ON burns the token budget before the answer; see providers.py.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = httpx.post(
        "https://inference.hetzner.com/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=body,
        timeout=180.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def probe(variant_name: str, text: str) -> tuple[bool | None, str]:
    """(fired, detail) for one call. `fired` is None when the case never
    reaches the LLM, so it can be excluded from scoring rather than
    miscounted as a silent negative."""
    numbered, bounds = build(text)
    if not bounds:
        return None, "no candidate boundary"
    prompt = VARIANTS[variant_name].format(paragraph=numbered, boundaries=bounds)
    try:
        content = call_model(prompt)
    except Exception as e:  # noqa: BLE001 - probe should survive one bad call
        return None, f"ERROR {type(e).__name__}: {e}"

    blocks = parse_block_response(
        content,
        fields=["NA_ZIN", "RELATIE", "HERSCHRIJVING", "UITLEG"],
        required="NA_ZIN",
    )
    kept = [
        b for b in blocks
        if (b.get("RELATIE") or "").strip().lower() in STRONG_RELATIONS
        and re.search(r"\d+", b.get("NA_ZIN", ""))
    ]
    if kept:
        b = kept[0]
        return True, f"{b.get('RELATIE')}: {b.get('HERSCHRIJVING', '')[:88]}"
    if blocks:
        relations = ",".join((b.get("RELATIE") or "?") for b in blocks)
        return False, f"discarded weak relation: {relations}"
    if "GEEN" in content.upper():
        return False, "GEEN"
    return False, f"UNPARSED {content[:60]!r}"


def run(variant_name: str, cases: list, reps: int, workers: int) -> dict:
    """{case_id: [(fired, detail), ...]} — reps calls per case, concurrently."""
    out: dict[str, list] = collections.defaultdict(list)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(probe, variant_name, text): cid
            for cid, text in cases
            for _ in range(reps)
        }
        for fut in cf.as_completed(futures):
            out[futures[fut]].append(fut.result())
    return out


def audit_examples() -> None:
    """Flag prompt-example vocabulary that collides with corpus text — the
    failure mode that cost corpus5 conn-4 4 of 5 fires."""
    examples = re.findall(r'^- "(.*?)"', BASE["user"], re.M)
    ex_words = {w.lower() for e in examples for w in re.findall(r"\w+", e) if len(w) > 4}
    hits = collections.defaultdict(list)
    for name in ("corpus4.json", "corpus5.json"):
        for item in json.load(open(os.path.join(HERE, name), encoding="utf-8"))["items"]:
            shared = ex_words & {
                w.lower() for w in re.findall(r"\w+", item["text"]) if len(w) > 4
            }
            if shared:
                hits[f"{name}:{item['id']}"] = sorted(shared)
    print(f"{len(examples)} worked examples in the connective prompt")
    if not hits:
        print("no vocabulary shared with corpus4/5 text")
        return
    print("shared vocabulary (single common nouns are usually fine; whole "
          "repeated phrases are what interfere):")
    for k, v in sorted(hits.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {k:24s} {', '.join(v)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="base", help="variant to score")
    ap.add_argument("--compare", default=None, help="second variant to A/B against")
    ap.add_argument("--reps", type=int, default=6,
                    help="calls per case (5-6 minimum to beat the noise)")
    ap.add_argument("--cases", default=None, help="comma-separated case ids")
    ap.add_argument("--corpora", default="corpus4.json,corpus5.json")
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent calls; >3 can brush the Hetzner rate cap")
    ap.add_argument("--audit", action="store_true",
                    help="report example/corpus vocabulary collisions and exit")
    args = ap.parse_args()

    if args.audit:
        audit_examples()
        return

    positives, negatives = load_cases(args.corpora.split(","))
    negatives += SYNTHETIC_NEGATIVES
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",")}
        positives = [c for c in positives if c[0] in wanted]
        negatives = [c for c in negatives if c[0] in wanted]
    if not positives and not negatives:
        raise SystemExit("no cases selected")

    names = [args.variant] + ([args.compare] if args.compare else [])
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n!r}; have: {', '.join(VARIANTS)}")

    runs = {n: run(n, positives + negatives, args.reps, args.workers) for n in names}

    header = f"{'case':22s}" + "".join(f"{n[:11]:>13s}" for n in names)
    print(f"\n{header}   want")
    totals = {n: [0, 0] for n in names}  # [good, scored]
    for group, cases, want_fire in (("positive", positives, True),
                                    ("negative", negatives, False)):
        print(f"-- {group}s " + "-" * (len(header) - len(group) - 4))
        for cid, _ in cases:
            cells = ""
            skipped = False
            for n in names:
                rows = runs[n][cid]
                scored = [f for f, _ in rows if f is not None]
                if not scored:
                    cells += f"{'skipped':>13s}"
                    skipped = True
                    continue
                fired = sum(scored)
                good = fired if want_fire else len(scored) - fired
                totals[n][0] += good
                totals[n][1] += len(scored)
                cells += f"{fired:>8d}/{len(scored):<4d}"
            if not skipped or len(names) > 1:
                print(f"{cid:22s}{cells}   {'FIRE' if want_fire else 'GEEN'}")

    print("\n-- totals " + "-" * (len(header) - 10))
    for n in names:
        good, scored = totals[n]
        pct = 100.0 * good / scored if scored else 0.0
        print(f"{n:22s} {good:>4d}/{scored:<5d} correct  ({pct:.0f}%)")
    if len(names) == 2:
        a, b = names
        delta = (totals[a][0] / max(totals[a][1], 1)) - (totals[b][0] / max(totals[b][1], 1))
        print(f"\n{a} - {b}: {delta * 100:+.1f} points "
              f"(at {args.reps} reps; treat <5 points as noise)")


if __name__ == "__main__":
    main()
