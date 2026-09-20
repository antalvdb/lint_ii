#!/usr/bin/env python3
"""Run the LiNT-II self-diagnosis corpus against the live box.

Posts each corpus item to /analyze, captures the suggestions, and writes
results.json incrementally (resumable: re-running skips items already done).
Each item is cache-busted with a per-run nonce so a run always re-analyses
(the box caches results by input text, persisted across restarts).

    python3 scripts/eval/run_eval.py            # full run
    python3 scripts/eval/run_eval.py --limit 5  # smoke test
    python3 scripts/eval/run_eval.py --fresh     # ignore existing results.json

The LLM-as-judge scoring (wrong/debatable/right + precision/recall) is done
separately from results.json; this script only gathers raw output and a
presence/absence summary.
"""
import argparse
import json
import os
import re
import time
import urllib.request

BASE = "https://lint-ii.valkuil.net"
HERE = os.path.dirname(__file__)

# Fields worth keeping per suggestion for judging. `model` distinguishes the
# two spelling passes (None = Hunspell, a model name = LLM) — eval set 4's
# spelling failures were mis-attributed to the LLM for lack of it.
KEEP = ("type", "sentence_index", "original_text", "suggested_text",
        "replacement_word", "relation", "list_intro", "list_items",
        "model", "error_category", "word")


def _post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _analyze(text):
    job = _post("/analyze", {"text": text, "max_suggestions": 50})["job_id"]
    for _ in range(90):
        with urllib.request.urlopen(BASE + "/analyze-result/" + job, timeout=30) as r:
            res = json.load(r)
        st = res.get("status")
        if st == "error":
            raise RuntimeError(res.get("error", "analyze error"))
        if st != "pending":
            return res["result"]
        time.sleep(2)
    raise TimeoutError("analysis did not finish in time")


def _slim(sug):
    out = {k: sug[k] for k in KEEP if k in sug}
    if sug.get("variants"):
        out["variants"] = [{"key": v.get("key"), "suggested_text": v.get("suggested_text")}
                           for v in sug["variants"]]
    return out


# --------------------------------------------------------------------------
# must_not scoring
#
# `must_not` was recorded in results.json from the start but NEVER SCORED --
# every guard it describes was checked by hand. That left the harness reporting
# only presence/absence, which conflates two different things on a negative
# item:
#
#   must_not = []            "nothing should fire here" (clean-*, good-*)
#   must_not = ['...']       "THIS must not fire; something else may be fine"
#
# Counting the second as a false positive understates the engine, and it is not
# a rare edge: on sets 2, 3 and 5 roughly half the FPs were items whose guard
# held while a different, legitimate suggestion type fired (shortlist-1 drawing
# max_sdl, conj-4 keeping ", maar " intact while simplifying a word, url-1
# splitting a sentence with the URL preserved). One of those, conj-4's
# invalidenplaatsen -> parkeerplaatsen voor gehandicapten, is a good suggestion
# that was being scored against us.
#
# So: presence/absence is kept unchanged for continuity with the cross-set
# table, and the guard check below is reported alongside it as the metric that
# actually asserts something.
# --------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+|[\w.+-]+@[\w-]+\.[\w.]+")


def guard_violations(must_not, produced):
    """Return [(rule, detail)] for each must_not rule actually breached."""
    viols = []
    for rule in must_not or []:
        r = rule.lower()
        if r == "enumeration":
            if any(s.get("type") == "enumeration" for s in produced):
                viols.append((rule, "enumeration suggestion produced"))
        elif "word_frequency" in r:
            bad = [s for s in produced if s.get("type") == "word_frequency"]
            if bad:
                viols.append((rule, f"word_frequency on {bad[0].get('word')!r}"))
        elif r == "suggest on non-prose":
            if produced:
                viols.append((rule, f"{len(produced)} suggestion(s) on non-prose"))
        elif r == "alter url":
            for s in produced:
                orig, new = s.get("original_text") or "", s.get("suggested_text") or ""
                for u in _URL_RE.findall(orig):
                    u = u.rstrip(".,;:)")
                    if u and u not in new:
                        viols.append((rule, f"{u} altered or dropped"))
        elif r.startswith("split"):
            # e.g. "split ', maar '" -> the conjunction must still join the clauses
            m = re.search(r"'(,[^']*)'", rule)
            conj = m.group(1) if m else None
            if conj:
                for s in produced:
                    orig, new = s.get("original_text") or "", s.get("suggested_text") or ""
                    if conj in orig and conj not in new:
                        viols.append((rule, f"{conj.strip()!r} no longer joins the clauses"))
        else:
            viols.append((rule, "UNCHECKED rule — no predicate implemented"))
    return viols



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--base", default=None,
                    help="API base URL; on the Strato box use http://127.0.0.1:8000 "
                         "(bypasses the nginx edge rate limits)")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    if args.base:
        global BASE
        BASE = args.base.rstrip("/")
    CORPUS, RESULTS = args.corpus, args.results

    corpus = json.load(open(CORPUS, encoding="utf-8"))["items"]
    if args.limit:
        corpus = corpus[:args.limit]

    results = {}
    if os.path.exists(RESULTS) and not args.fresh:
        results = json.load(open(RESULTS, encoding="utf-8")).get("results", {})

    nonce = int(time.time())
    done = 0
    for item in corpus:
        iid = item["id"]
        if iid in results and not results[iid].get("error"):
            continue
        text = item["text"] + f"\n\nTestref {nonce}."
        rec = {"should_suggest": item["should_suggest"],
               "phenomena": item.get("phenomena", []),
               "must_not": item.get("must_not", []),
               "text": item["text"]}
        try:
            data = _analyze(text)
            sugs = data.get("suggestions", {}).get("suggestions", [])
            # drop suggestions on the cache-busting nonce block
            sugs = [s for s in sugs if "Testref" not in (s.get("original_text") or "")]
            rec["produced"] = [_slim(s) for s in sugs]
            rec["types"] = sorted({s.get("type") for s in sugs})
            rec["error"] = None
        except Exception as e:
            rec["produced"], rec["types"], rec["error"] = [], [], str(e)
        results[iid] = rec
        done += 1
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump({"base": BASE, "nonce": nonce, "results": results}, f,
                      ensure_ascii=False, indent=1)
        print(f"[{done}] {iid}: {rec['types'] or ('ERROR: ' + rec['error'] if rec['error'] else 'none')}",
              flush=True)
        time.sleep(0.4)

    # Presence/absence summary (precision/recall of "produced any suggestion").
    tp = fp = fn = tn = 0
    for item in corpus:
        r = results.get(item["id"])
        if not r or r.get("error"):
            continue
        produced = bool(r["produced"])
        want = item["should_suggest"]
        if want and produced: tp += 1
        elif want and not produced: fn += 1
        elif not want and produced: fp += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec_ = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"\nPresence/absence: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={prec:.2f}  Recall={rec_:.2f}   (legacy metric — comparable"
          f" with the cross-set table)")

    # Guard check: the metric that actually asserts something. Also split the
    # negatives so a "false positive" on a guarded item is not confused with
    # one on an item that should simply be silent.
    violations = []
    silent_fired = guarded_fired = 0
    for item in corpus:
        r = results.get(item["id"])
        if not r or r.get("error"):
            continue
        produced = r["produced"]
        for rule, detail in guard_violations(item.get("must_not"), produced):
            violations.append((item["id"], rule, detail))
        if not item["should_suggest"] and produced:
            if item.get("must_not"):
                guarded_fired += 1
            else:
                silent_fired += 1

    print(f"\nGuard violations (must_not actually breached): {len(violations)}")
    for iid, rule, detail in violations:
        print(f"  {iid}: [{rule}] {detail}")
    print(f"\nNegatives that produced something, split by kind:")
    print(f"  must_not=[]   (should be silent) : {silent_fired}"
          f"   <- genuine precision concern")
    print(f"  guarded items (other type fired) : {guarded_fired}"
          f"   <- not a defect if the guard held; see README")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
