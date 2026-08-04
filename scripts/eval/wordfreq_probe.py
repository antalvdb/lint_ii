#!/usr/bin/env python3
"""Probe for backlog item 2 — meaning-changing word_frequency swaps.

Sends the REAL word_frequency_bundle prompt as single-item bundles (how the
corpus cases arose), and classifies the returned VERVANGING against
hand-labelled accept/reject sets. Every distinct output is printed, so a new
bad swap cannot hide behind the lists.

    python3 wordfreq_probe.py --reps 12                  # score the live prompt
    python3 wordfreq_probe.py --variant mine --compare base --reps 12
    python3 wordfreq_probe.py --group control --reps 4   # must-still-simplify

TWO GROUPS, and you need BOTH:
  hard    - 11 reproducible bad swaps from the eval runs. Lower BAD is better.
  control - 10 words a simplification MUST still fire on. Here a KEPT/
            ONGEWIJZIGD answer is a FAILURE. Without this group a variant can
            score beautifully by simply refusing to do its job.

RULES THIS PROBE EARNED (all learned by getting them wrong first):

1. 12 reps minimum for an aggregate claim. At 5 reps two runs of an IDENTICAL
   prompt scored 55% and 42% BAD — a 13-point swing, larger than the effect
   being measured. 5 reps is enough for a single per-case verdict, not for a
   total.
2. Do NOT use test-set or corpus words as prompt examples. The first variant
   scored 65%->22% but four of its examples were four of its own test cases;
   held-out it was 49%->29%. See the lexical-leak rule in CLAUDE.md.
3. Prompt examples move WOBBLY cases and cannot touch CONFIDENT ones. Every
   case failing 12/12 at base still failed 12/12 after the winning variant;
   the whole gain came from partially-failing cases. If a swap is deterministic
   at base, a prompt will not fix it — see swap_judge_probe.py instead.

SCOPE: this exercises the BUNDLED path only. A sentence whose word_frequency
trigger gets folded into a consolidated sentence_rewrite takes a different
prompt, which carries none of this guidance and is not measured here.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

from lint_ii.llm.prompts import PROMPT_TEMPLATES, parse_block_response  # noqa: E402

BUNDLE = PROMPT_TEMPLATES["word_frequency_bundle"]


def norm(s: str) -> str:
    return re.sub(r"[^a-zà-ÿ ]", "", (s or "").lower().strip())


# (id, word, zipf, context, GOOD replacements, BAD replacements)
HARD = [
    ("monumentale", "monumentale", 2.85,
     "Het monumentale orgel wordt door een gespecialiseerde restaurator pijp voor pijp schoongemaakt.",
     {"historische", "beschermde", "eeuwenoude", "historisch waardevolle"},
     {"groot", "grote", "oude", "waardevol", "waardevolle", "mooie", "indrukwekkende"}),
    ("koeling", "koeling", 2.96,
     "Het is door de keuringsdienst vastgesteld dat bij twee kramen op de markt de koeling van verse vis niet op orde was.",
     {"koeling", "gekoelde opslag", "temperatuur", "koelsysteem", "koelinstallatie"},
     {"koelkast", "vriezer", "koelbox", "ijskast"}),
    ("gewaande", "gewaande", 2.60,
     "De verloren gewaande brieven zijn door een medewerker van het archief bij toeval teruggevonden.",
     {"gedachte", "veronderstelde", "gewaande"},
     {"vermeende", "verdwenen", "zoekgeraakte", "kwijtgeraakte"}),
    ("verharding", "verharding", 2.71,
     "Het traject leidde tot een verzakelijking van de verhoudingen en een verharding van de standpunten.",
     {"verharding", "harder worden", "het harder worden", "verscherping", "aanscherping"},
     {"vastberadenheid", "versteviging", "verstarring", "verharden"}),
    ("structureel", "structureel", 3.12,
     "Het is door de arbeidsinspectie vastgesteld dat er in het sorteercentrum structureel te zware pakketten door één persoon werden getild.",
     {"stelselmatig", "voortdurend", "systematisch", "steeds weer", "regelmatig"},
     {"altijd", "vaak", "soms", "structuur"}),
    ("insinuaties", "insinuaties", 2.55,
     "De advocaat sprak van kwalijke insinuaties aan het adres van haar cliënt.",
     {"verdachtmakingen", "beschuldigingen", "aantijgingen", "toespelingen"},
     {"suggesties", "opmerkingen", "ideeën", "voorstellen"}),
    ("conservator", "conservator", 2.90,
     "De conservator legde tijdens de rondleiding uit dat het schilderij na een grondige technische analyse aan een beroemde meester kan worden toegeschreven.",
     {"conservator", "museummedewerker", "beheerder van de collectie", "curator", "collectiebeheerder"},
     {"bewaarder", "suppoost", "bewaker", "opzichter"}),
    ("ambivalent", "ambivalent", 2.40,
     "De wethouder toonde zich ambivalent over de komst van het distributiecentrum.",
     {"verdeeld", "aarzelend", "weifelend", "besluiteloos", "twijfelend"},
     {"twijfelachtig", "onzeker", "negatief", "positief", "onduidelijk"}),
    ("notoire", "notoire", 2.65,
     "De naburige camping staat in de regio bekend als een notoire bron van nachtelijke overlast.",
     {"beruchte", "berucht"},
     {"bekende", "bekend", "grote", "belangrijke"}),
    ("verwaarloosde", "verwaarloosde", 2.95,
     "Het is door de dierenbescherming bevestigd dat de verwaarloosde pony's inmiddels bij een gastgezin zijn ondergebracht.",
     {"verwaarloosde", "slecht verzorgde", "mishandelde", "veronachtzaamde"},
     {"vergeten", "achtergelaten", "zieke", "arme"}),
    ("reder", "reder", 2.70,
     "De reder noemde de gevraagde liggelden ronduit exorbitant.",
     {"rederij", "scheepseigenaar", "eigenaar van het schip", "reder"},
     {"eigenaar", "kapitein", "schipper", "baas"}),
]

# KEPT here is a FAILURE. Empty GOOD set = compound split; any multiword answer
# counts as correct.
CONTROL = [
    ("behelst", "behelst", 3.10,
     "De voorbereiding op de vogeltelling behelst het uitdelen van telformulieren en het instrueren van de vrijwilligers.",
     {"omvat", "bestaat uit", "houdt in", "betekent", "bevat"}, set()),
    ("clandestiene", "clandestiene", 2.45,
     "De douane stuitte in de haven op een clandestiene handel in beschermde planten.",
     {"illegale", "verboden", "stiekeme", "geheime", "illegaal"}, set()),
    ("futiele", "futiele", 2.50,
     "De rechter deed de aangevoerde bezwaren af als futiele details.",
     {"onbelangrijke", "kleine", "onbeduidende", "nietige", "triviale"}, set()),
    ("gepikeerd", "gepikeerd", 2.60,
     "De burgemeester reageerde zichtbaar gepikeerd op de kritische vragen van de raad.",
     {"boos", "geïrriteerd", "geraakt", "beledigd", "verontwaardigd"}, set()),
    ("rigide", "rigide", 2.75,
     "De inspectie verweet de instelling een rigide toepassing van de huisregels.",
     {"strenge", "streng", "starre", "star", "strikte", "strikt", "onbuigzame"}, set()),
    ("precair", "precair", 2.68,
     "De financiering van het jeugdhonk blijft volgens de wethouder uiterst precair.",
     {"onzeker", "wankel", "kwetsbaar", "instabiel", "moeilijk"}, set()),
    ("reprimande", "reprimande", 2.40,
     "De rechter gaf de advocaat een stevige reprimande wegens zijn late stukken.",
     {"waarschuwing", "berisping", "standje", "uitbrander", "terechtwijzing"}, set()),
    ("stringentere", "stringentere", 2.55,
     "De toezichthouder hanteert sinds dit jaar aanzienlijk stringentere regels voor kleine fondsen.",
     {"strengere", "strenger", "striktere", "scherpere"}, set()),
    ("bodemsanering", "bodemsaneringsprogramma", 1.36,
     "Het bodemsaneringsprogramma voor het voormalige fabrieksterrein duurt zeker vier jaar.",
     set(), set()),
    ("drinkwaterleiding", "drinkwatertransportleiding", 1.36,
     "De drinkwatertransportleiding onder de rivier wordt in fasen vervangen.",
     set(), set()),
]


def variant(old: str, new: str, *, name: str) -> str:
    """BASE with `old` -> `new`; dies if the anchor is gone or already patched.

    Without this a prompts.py edit silently turns a candidate back into base
    (or double-inserts the block) and you A/B a prompt against itself.
    """
    text = BUNDLE["user"]
    if new.strip() and new.strip() in text:
        raise SystemExit(
            f"variant {name!r}: its block is ALREADY in the live prompt. It was "
            f"probably shipped; comparing it to base would compare base to base."
        )
    if old not in text:
        raise SystemExit(
            f"variant {name!r}: anchor not found — re-anchor against the live "
            f"prompt.\n  anchor: {old[:80]!r}..."
        )
    return text.replace(old, new)


VARIANTS: dict[str, str] = {"base": BUNDLE["user"]}
# Add candidates here, e.g.:
#   VARIANTS["stricter"] = variant(ANCHOR, MY_BLOCK + ANCHOR, name="stricter")
# The meaning-preservation block from 3992b09 is already live, so it is not
# redefined here — variant() would reject it.


def call(prompt: str) -> str:
    import httpx
    key = os.environ.get("HETZNER_API_KEY")
    if not key:
        raise SystemExit("HETZNER_API_KEY not set (box: /etc/lint-ii/lint-ii.env)")
    r = httpx.post(
        "https://inference.hetzner.com/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.environ.get("LINT_II_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8"),
            "messages": [
                {"role": "system", "content": BUNDLE["system"]},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.environ.get("LINT_II_HETZNER_TEMPERATURE", "0.3")),
            "max_tokens": 1200,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def probe(variant_name: str, case, control: bool):
    cid, word, zipf, ctx, good, bad = case
    items = f'1. WOORD: "{word}" (frequentie {zipf:.2f})\n   FRAGMENT: "{ctx}"'
    try:
        content = call(VARIANTS[variant_name].format(n_items=1, items=items))
    except Exception as e:  # noqa: BLE001
        return cid, "ERROR", f"{type(e).__name__}"
    blocks = parse_block_response(
        content, fields=["NUMMER", "VERVANGING", "UITLEG", "HERSCHRIJVING"], required="NUMMER")
    if not blocks:
        return cid, "UNPARSED", content[:50]
    v = norm(blocks[0].get("VERVANGING"))
    if not v:
        return cid, "EMPTY", ""
    if v == "ongewijzigd" or v == norm(word):
        return cid, ("REFUSED" if control else "KEPT"), v
    if v in {norm(g) for g in good}:
        return cid, "GOOD", v
    if v in {norm(b) for b in bad}:
        return cid, "BAD", v
    if control and not good and " " in v:
        return cid, "GOOD", v          # compound split: any paraphrase is fine
    return cid, "UNJUDGED", v


def run(variant_name: str, cases, reps: int, control: bool, workers: int):
    out = collections.defaultdict(list)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(probe, variant_name, c, control): c[0]
                for c in cases for _ in range(reps)}
        for f in cf.as_completed(futs):
            cid, verdict, val = f.result()
            out[cid].append((verdict, val))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--reps", type=int, default=12,
                    help="12+ for an aggregate claim; 5 swings by 13 points")
    ap.add_argument("--group", choices=["hard", "control"], default="hard")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    cases = HARD if args.group == "hard" else CONTROL
    control = args.group == "control"
    names = [args.variant] + ([args.compare] if args.compare else [])
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n!r}; have: {', '.join(VARIANTS)}")

    runs = {n: run(n, cases, args.reps, control, args.workers) for n in names}
    key = "REFUSED" if control else "BAD"
    print(f"\n=== {args.group} group, {args.reps} reps ===")
    print(f"{'case':16s}" + "".join(f"{n[:12]:>14s}" for n in names) + f"   ({key} count)")
    tot = {n: 0 for n in names}
    for c in cases:
        cid = c[0]
        cells = ""
        for n in names:
            k = collections.Counter(v for v, _ in runs[n][cid])[key]
            tot[n] += k
            cells += f"{k:>9d}/{args.reps:<4d}"
        print(f"{cid:16s}{cells}")
        for n in names:
            top = collections.Counter(f"{v}:{x}" for v, x in runs[n][cid]).most_common(2)
            print(f"      {n:10s} " + "  ".join(f"{c_}x {k}" for k, c_ in top))
    n_obs = args.reps * len(cases)
    print(f"\n--- totals ({key} is the failure) ---")
    for n in names:
        print(f"  {n:14s} {tot[n]:3d}/{n_obs}  ({100*tot[n]/n_obs:.0f}%)")
    if len(names) == 2:
        a, b = names
        print(f"\n  {a} - {b}: {100*(tot[a]-tot[b])/n_obs:+.0f} points "
              f"(at {args.reps} reps; <8 points is not a result)")


if __name__ == "__main__":
    main()
