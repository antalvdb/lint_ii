#!/usr/bin/env python3
"""Can Qwen judge its OWN word swaps? Feasibility probe for a verification pass.

Backlog item 2 leaves a class of meaning-changing swaps that prompt work cannot
touch (see wordfreq_probe.py rule 3). The remaining option is a verification
pass: a second call that rejects a bad swap before it reaches the user. This
probe answers whether that can work AT ALL, and at what operating point.

    python3 swap_judge_probe.py --judge calibrated --reps 5
    python3 swap_judge_probe.py --judge strict --compare calibrated

WHAT IT ALREADY SETTLED (2026-08-04):

- The judge does NOT share the generator's error. Qwen produces
  monumentale->groot 12/12, yet asked directly it rejects that swap 5/5. The
  obvious objection to a same-model verification pass is simply false.
- The whole problem is CALIBRATION, and it is severe:
      strict     ("precies hetzelfde?")  100% detection / 60% FALSE ALARMS
      calibrated ("zet dit de lezer op het verkeerde been?")
                                          63% detection /  0% false alarms
  A judge is only useful with a low false-alarm rate: every false alarm is a
  legitimate simplification destroyed, and those are the product.
- Discount the taught cases. `calibrated` names koeling/notoire (as FOUT) and
  reprimande/clandestiene/gepikeerd (as GOED) in its own prompt, so its honest
  held-out detection is 45%, not 63%. TAUGHT below marks them and the summary
  reports held-out separately. Do not remove that split.

NOT YET MEASURED, and required before building this: cost and latency. This is
one extra call per word_frequency suggestion, and the eval runs produce ~45 of
those per 100-item set against an output-bound rate limit.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import os
import re
import sys

# (word, replacement, sentence, should_reject)
CASES = [
    # confident-wrong: the generator emits these 12/12 and no prompt stops it
    ("monumentale", "grote",
     "Het monumentale orgel wordt door een gespecialiseerde restaurator pijp voor pijp schoongemaakt.", True),
    ("gewaande", "vermeende",
     "De verloren gewaande brieven zijn door een medewerker van het archief bij toeval teruggevonden.", True),
    ("insinuaties", "suggesties",
     "De advocaat sprak van kwalijke insinuaties aan het adres van haar cliënt.", True),
    ("notoire", "bekende",
     "De naburige camping staat in de regio bekend als een notoire bron van nachtelijke overlast.", True),
    # partially-failing
    ("koeling", "koelkast",
     "Bij twee kramen op de markt was de koeling van verse vis niet op orde.", True),
    ("conservator", "bewaarder",
     "De conservator legde tijdens de rondleiding uit hoe het schilderij was toegeschreven.", True),
    # CONTROLS: legitimate simplifications; rejecting these is the costly error
    ("clandestiene", "illegale",
     "De douane stuitte in de haven op een clandestiene handel in beschermde planten.", False),
    ("futiele", "onbelangrijke",
     "De rechter deed de aangevoerde bezwaren af als futiele details.", False),
    ("gepikeerd", "boos",
     "De burgemeester reageerde zichtbaar gepikeerd op de kritische vragen van de raad.", False),
    ("reprimande", "waarschuwing",
     "De rechter gaf de advocaat een stevige reprimande wegens zijn late stukken.", False),
    ("stringentere", "strengere",
     "De toezichthouder hanteert sinds dit jaar aanzienlijk stringentere regels voor kleine fondsen.", False),
    ("precair", "onzeker",
     "De financiering van het jeugdhonk blijft volgens de wethouder uiterst precair.", False),
]

# Words named in the `calibrated` prompt. Their scores are inflated; the
# summary reports held-out separately so the honest number stays visible.
TAUGHT = {"koeling", "notoire", "reprimande", "clandestiene", "gepikeerd"}
# `production` teaches none of these; its held-out figure equals its overall one.
TAUGHT_BY = {"calibrated": TAUGHT, "strict": set(), "production": set(), "production_v2": set(), "production_v3": set()}

JUDGES = {
    # Rejects nearly everything: "exactly the same?" is a stricter question
    # than the product needs. Kept as the cautionary baseline.
    "strict": '''Je controleert of een woordvervanging de betekenis van een zin intact laat.

Zin: "{sent}"
Voorgestelde vervanging: "{word}" wordt "{repl}"

Betekent de zin met "{repl}" precies hetzelfde als met "{word}"?

Antwoord met exact een van deze twee regels:
OORDEEL: BEHOUDEN
OORDEEL: VERANDERD''',

    # Production candidate. Same calibration as `calibrated`, but its examples
    # are drawn from outside every corpus AND outside this test set, so all 12
    # cases below stay held-out. `calibrated` names koeling/notoire/reprimande/
    # clandestiene/gepikeerd in its own prompt and its 63% is inflated; this one
    # can be read at face value.
    "production": '''Je beoordeelt of een woordvervanging in een tekst voor gewone lezers acceptabel is.

Zin: "{sent}"
Vervanging: "{word}" wordt "{repl}"

Het doel is de tekst MAKKELIJKER te maken. Een eenvoudiger woord met dezelfde strekking is GOED, ook als het net iets algemener of gewoner klinkt; kleine stijlverschillen zijn geen bezwaar.

Keur alleen AF als de vervanging de lezer op het verkeerde been zet:
- het wordt een ander ding (een handeling wordt een apparaat)
- de gevoelswaarde draait om (negatief wordt neutraal of positief)
- er wordt iets specifieks beweerd dat er niet stond

Voorbeelden van GOED: "terstond" -> "meteen"; "gaarne" -> "graag".
Voorbeelden van FOUT: "verhitting" -> "oven" (handeling wordt apparaat); "eigenzinnige" -> "bijzondere" (het oordeel verdwijnt).

Antwoord met exact een van deze twee regels:
OORDEEL: GOED
OORDEEL: FOUT''',
    # production_v2: adds the denominalization rule. The first production judge
    # rejected "verlaging -> minder" and "afname -> minder" on a full eval run
    # -- both are the abstract-noun class the pipeline EXISTS to simplify, so
    # that was the judge attacking the product. The rule is taught by pattern
    # with clear vocabulary (daling/wachttijd/lagere), never by naming
    # verlaging/afname/minder, so those stay held-out.
    "production_v2": '''Je beoordeelt of een woordvervanging in een tekst voor gewone lezers acceptabel is.

Zin: "{sent}"
Vervanging: "{word}" wordt "{repl}"

Het doel is de tekst MAKKELIJKER te maken. Een eenvoudiger woord met dezelfde strekking is GOED, ook als het net iets algemener of gewoner klinkt; kleine stijlverschillen zijn geen bezwaar.

Een omslachtige naamwoordconstructie vervangen door een gewoner woord, een werkwoord of een bijvoeglijk naamwoord is juist GOED — dat is het doel van deze tekstverbetering. Bijvoorbeeld "een daling van de wachttijd" -> "een lagere wachttijd".

Keur alleen AF als de vervanging de lezer op het verkeerde been zet:
- het wordt een ander ding (een handeling wordt een apparaat)
- de gevoelswaarde draait om (negatief wordt neutraal of positief)
- er wordt iets specifieks beweerd dat er niet stond

Voorbeelden van GOED: "terstond" -> "meteen"; "gaarne" -> "graag".
Voorbeelden van FOUT: "verhitting" -> "oven" (handeling wordt apparaat); "eigenzinnige" -> "bijzondere" (het oordeel verdwijnt).

Antwoord met exact een van deze twee regels:
OORDEEL: GOED
OORDEEL: FOUT''',
    # production_v3: production + a NARROW exemption. v2's blanket "replacing a
    # clumsy noun construction is good" halved detection (47% -> 25%) because it
    # licenses almost anything. This names only the change-nominalization shape
    # (a noun expressing an increase/decrease -> a plain wording of the same
    # increase/decrease), taught with "stijging -> meer leden" so the actual
    # false-alarm pairs (verlaging->minder, afname->minder) stay held-out.
    "production_v3": '''Je beoordeelt of een woordvervanging in een tekst voor gewone lezers acceptabel is.

Zin: "{sent}"
Vervanging: "{word}" wordt "{repl}"

Het doel is de tekst MAKKELIJKER te maken. Een eenvoudiger woord met dezelfde strekking is GOED, ook als het net iets algemener of gewoner klinkt; kleine stijlverschillen zijn geen bezwaar.

Een naamwoord dat een toename of afname uitdrukt mag worden vervangen door een gewone formulering van diezelfde toename of afname; de betekenis blijft dan gelijk. Bijvoorbeeld "een stijging van het aantal leden" -> "meer leden". Dat is GOED.

Keur alleen AF als de vervanging de lezer op het verkeerde been zet:
- het wordt een ander ding (een handeling wordt een apparaat)
- de gevoelswaarde draait om (negatief wordt neutraal of positief)
- er wordt iets specifieks beweerd dat er niet stond

Voorbeelden van GOED: "terstond" -> "meteen"; "gaarne" -> "graag".
Voorbeelden van FOUT: "verhitting" -> "oven" (handeling wordt apparaat); "eigenzinnige" -> "bijzondere" (het oordeel verdwijnt).

Antwoord met exact een van deze twee regels:
OORDEEL: GOED
OORDEEL: FOUT''',
    "calibrated": '''Je beoordeelt of een woordvervanging in een tekst voor gewone lezers acceptabel is.

Zin: "{sent}"
Vervanging: "{word}" wordt "{repl}"

Het doel is de tekst MAKKELIJKER te maken. Een eenvoudiger woord met dezelfde strekking is GOED, ook als het net iets algemener of gewoner klinkt; kleine stijlverschillen zijn geen bezwaar.

Keur alleen AF als de vervanging de lezer op het verkeerde been zet:
- het wordt een ander ding (een handeling wordt een apparaat)
- de gevoelswaarde draait om (negatief wordt neutraal of positief)
- er wordt iets specifieks beweerd dat er niet stond

Voorbeelden van GOED: "reprimande" -> "waarschuwing"; "clandestiene" -> "illegale"; "gepikeerd" -> "boos".
Voorbeelden van FOUT: "koeling" -> "koelkast"; "notoire" -> "bekende".

Antwoord met exact een van deze twee regels:
OORDEEL: GOED
OORDEEL: FOUT''',
}


def call(prompt: str) -> str:
    import httpx
    key = os.environ.get("HETZNER_API_KEY")
    if not key:
        raise SystemExit("HETZNER_API_KEY not set (box: /etc/lint-ii/lint-ii.env)")
    r = httpx.post(
        "https://inference.hetzner.com/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.environ.get("LINT_II_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8"),
              "messages": [{"role": "user", "content": prompt}],
              "temperature": float(os.environ.get("LINT_II_HETZNER_TEMPERATURE", "0.3")),
              "max_tokens": 200,
              "chat_template_kwargs": {"enable_thinking": False}},
        timeout=120.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def probe(judge: str, case):
    word, repl, sent, _ = case
    try:
        out = call(JUDGES[judge].format(sent=sent, word=word, repl=repl)).upper()
    except Exception as e:  # noqa: BLE001
        return (word, repl), "ERROR"
    if "VERANDERD" in out or re.search(r"OORDEEL:\s*FOUT", out):
        return (word, repl), "REJECT"
    if "BEHOUDEN" in out or re.search(r"OORDEEL:\s*GOED", out):
        return (word, repl), "ACCEPT"
    return (word, repl), "UNPARSED"


def score(judge: str, reps: int, workers: int):
    res = collections.defaultdict(list)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(probe, judge, c): (c[0], c[1]) for c in CASES for _ in range(reps)}
        for f in cf.as_completed(futs):
            k, v = f.result()
            res[k].append(v)
    return res


def report(judge: str, res, reps: int) -> None:
    print(f"\n=== judge {judge!r}, {reps} reps ===")
    det = miss = fa = ok = 0
    det_h = tot_h = 0
    for word, repl, _s, should_reject in CASES:
        c = collections.Counter(res[(word, repl)])
        taught = " [TAUGHT]" if word in TAUGHT_BY.get(judge, TAUGHT) else ""
        if should_reject:
            det += c["REJECT"]; miss += reps - c["REJECT"]
            if not taught:
                det_h += c["REJECT"]; tot_h += reps
            verdict = "detected" if c["REJECT"] > reps / 2 else "MISSED  "
        else:
            fa += c["REJECT"]; ok += c["ACCEPT"]
            verdict = "ok      " if c["ACCEPT"] > reps / 2 else "FALSE ALARM"
        print(f"  {'REJECT' if should_reject else 'accept':6s} "
              f"{word + ' -> ' + repl:32s} reject={c['REJECT']:2d} accept={c['ACCEPT']:2d}  {verdict}{taught}")
    nb = sum(1 for c in CASES if c[3]) * reps
    ng = sum(1 for c in CASES if not c[3]) * reps
    print(f"\n  detection      {det}/{nb} ({100*det/nb:.0f}%)"
          + (f"   held-out {det_h}/{tot_h} ({100*det_h/tot_h:.0f}%)" if tot_h else ""))
    print(f"  FALSE ALARMS   {fa}/{ng} ({100*fa/ng:.0f}%)   <- the number that decides usability")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="calibrated", choices=list(JUDGES))
    ap.add_argument("--compare", default=None, choices=list(JUDGES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    for j in [args.judge] + ([args.compare] if args.compare else []):
        report(j, score(j, args.reps, args.workers), args.reps)


if __name__ == "__main__":
    main()
