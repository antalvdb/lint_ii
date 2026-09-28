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

Kleijn mode (backlog item 8) runs the LIN cloze texts instead of a corpus and
scores the run against the ground truth from kleijn_truth.py:

    python3 scripts/eval/run_eval.py --kleijn --owners-ok \
        --base http://127.0.0.1:8000 [--versions moei,mak] [--limit 2]

The texts are likely copyrighted and a run sends them to the LLM provider, so
the mode refuses without --owners-ok (the text owners' permission, which is
Antal's to obtain). Its results quote the texts and default to the gitignored
private/kleijn/results.json.
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
        "model", "error_category", "word", "component_types", "merges_sentences")

KLEIJN_DIR = os.path.join(HERE, "private", "kleijn")


def _post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _analyze(text, fmt="text"):
    job = _post("/analyze", {"text": text, "max_suggestions": 50, "format": fmt})["job_id"]
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


# --------------------------------------------------------------------------
# Provider-error gate
#
# Every pass is FAIL-OPEN: a provider call that fails (429 rate limit, 5xx)
# silently becomes "no suggestion", which this runner then scores as a false
# negative. The eval cannot see that from the API response -- only the
# service's log records it. So the runner reads the log around each item.
#
# Why this is automatic rather than a checklist step: the manual gate in the
# README counted only 500s, and on 2026-09-11 set 3 ran through ~30 429s with
# a clean 500 count. Six of its seven "misses" (incl. all three connective
# ones) produced suggestions when re-tested, so the reported 0.89 recall -- and
# the conclusion that recall/connective was the engine's weak axis -- was a
# rate-limit artifact. 429s rose ~40x in Sep 2026, so this is now the common
# failure mode, not an edge case.
#
# The httpx line below is logged once per provider call, so counting it gives
# an exact per-item error count. Other traffic on the service (testers, cache
# warming) can be misattributed to an item; the cost of that is one extra
# retry, which is acceptable.
# --------------------------------------------------------------------------

_PROVIDER_ERR_RE = re.compile(
    r'HTTP Request: POST \S+/chat/completions "HTTP/1\.1 (429|5\d\d)'
)
# A call that never gets an HTTP answer (connection refused, timeout, network
# failure) logs no request line at all, only a traceback ending in the httpx
# exception. Without this the gate read an unreachable provider as CLEAN: a
# Kleijn smoke test on 2026-09-27 had every LLM call refused and still passed.
_TRANSPORT_ERR_RE = re.compile(r"^httpx\.(\w+(?:Error|Timeout))\b", re.M)


def _log_size(path):
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _provider_errors_since(path, offset):
    """(count, {status: n}) of provider-call failures logged after `offset`."""
    if path is None or offset is None:
        return 0, {}
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return 0, {}
    counts = {}
    for rx in (_PROVIDER_ERR_RE, _TRANSPORT_ERR_RE):
        for m in rx.finditer(chunk):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return sum(counts.values()), counts


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


def kleijn_items(text_dir, versions):
    """The Kleijn texts as run items: [{"id": "T01_moei", "text": ...}], in
    text order. The ground truth is keyed by the same ids."""
    items = []
    for name in sorted(os.listdir(text_dir)):
        m = re.fullmatch(r"(T\d\d)_(mak|moei)\.txt", name)
        if m and m.group(2) in versions:
            with open(os.path.join(text_dir, name), encoding="utf-8-sig") as f:
                items.append({"id": f"{m.group(1)}_{m.group(2)}", "text": f.read()})
    return items


def kleijn_report(results, truth_path):
    """Score a Kleijn run against the ground truth (kleijn_truth.score)."""
    import sys
    sys.path.insert(0, HERE)
    from kleijn_truth import score
    with open(truth_path, encoding="utf-8") as f:
        truth = json.load(f)
    runs = {iid: r["produced"] for iid, r in results.items() if not r.get("error")}
    out = score(truth, runs)
    wf, wo, cn = out["word_frequency"], out["word_order"], out["connectives"]
    pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else f"{a}/0"
    print("\n== Kleijn ground truth (difficult versions)")
    print(f"  rarer words   targeted {pct(wf['targeted'], wf['hard_words'])}, "
          f"easy original restored {pct(wf['restored'], wf['hard_words'])}, "
          f"hard word gone from some suggestion {pct(wf['changed_any'], wf['hard_words'])}")
    print(f"  word order    rewrite on a reordered sentence {pct(wo['fired'], wo['manipulated'])}; "
          f"on the ones our parser measures {pct(wo['measurable_fired'], wo['measurable'])}; "
          f"on untouched sentences {pct(wo['other_fired'], wo['other'])}")
    print(f"  connectives   merge at a removal {pct(cn['found'], cn['expected'])}, "
          f"same relation {pct(cn['relation_match'], cn['expected'])}")
    if out["per_sentence"]:
        print("  suggestions per sentence (over-editing check: easy should be lower):")
        for key, (n, sents) in sorted(out["per_sentence"].items()):
            print(f"    {key:22s} {n / sents:.2f}  ({n} on {sents} sentences)")
    return out


def validity_report(results, ids, log_path, provider_log, retries):
    """Was any scored item still contaminated by provider errors after its
    retries? Such an item's "miss" may be the provider, not the engine."""
    scored = [results[i] for i in ids if results.get(i) and not results[i].get("error")]
    dirty = [i for i in ids if results.get(i) and results[i].get("provider_errors")]
    retried = sum(1 for r in scored if (r.get("attempts") or 1) > 1)
    unchecked = [r for r in scored if not r.get("validity_source")]
    print()
    if unchecked and log_path is None:
        print(f"VALIDITY: NOT CHECKED for {len(unchecked)} item(s) — the service "
              f"reported no provider_failures and the log is unreadable ({provider_log}). "
              "Provider failures would silently read as false negatives.")
    elif dirty:
        print(f"VALIDITY: CONTAMINATED — {len(dirty)} item(s) still saw provider "
              f"errors after {retries} retries: {', '.join(dirty)}")
        print("  Their outcome may reflect the provider, not the engine. "
              "Re-run with the same --results (resumable) or treat as void.")
    else:
        print(f"VALIDITY: CLEAN — no provider errors on any scored item "
              f"({retried} item(s) needed a retry to get there).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--base", default=None,
                    help="API base URL; on the Strato box use http://127.0.0.1:8000 "
                         "(bypasses the nginx edge rate limits)")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus.json"))
    ap.add_argument("--results", default=None,
                    help="default: results.json, or private/kleijn/results.json with --kleijn")
    ap.add_argument("--kleijn", action="store_true",
                    help="run the Kleijn LIN texts (private/kleijn/texts) instead of a corpus")
    ap.add_argument("--versions", default="moei",
                    help="Kleijn versions to run: moei, mak, or moei,mak")
    ap.add_argument("--owners-ok", action="store_true",
                    help="confirm the Kleijn text owners agreed to sending the texts "
                         "to the LLM provider; the Kleijn mode refuses without it")
    ap.add_argument("--provider-log", default="/var/log/lint-ii/app.log",
                    help="service log to scan for provider 429/5xx per item "
                         "(box side). If unreadable, validity is NOT checked "
                         "and the summary says so.")
    ap.add_argument("--retries", type=int, default=2,
                    help="re-run an item up to N times if provider errors were "
                         "logged during it")
    args = ap.parse_args()
    if args.base:
        global BASE
        BASE = args.base.rstrip("/")
    if args.kleijn:
        if not args.owners_ok:
            raise SystemExit(
                "Kleijn mode sends the texts to the LLM provider. The texts are likely "
                "copyrighted: run it only once the owners have agreed, and then pass "
                "--owners-ok. (Local scoring needs no run: see kleijn_lint.py.)")
        versions = {v.strip() for v in args.versions.split(",") if v.strip()}
        if not versions or not versions <= {"mak", "moei"}:
            raise SystemExit("--versions must be moei, mak or moei,mak")
        corpus = kleijn_items(os.path.join(KLEIJN_DIR, "texts"), versions)
        RESULTS = args.results or os.path.join(KLEIJN_DIR, "results.json")
        fmt = "markdown"  # sentence numbers must match the ground truth's
    else:
        corpus = json.load(open(args.corpus, encoding="utf-8"))["items"]
        RESULTS = args.results or os.path.join(HERE, "results.json")
        fmt = "text"
    if args.limit:
        corpus = corpus[:args.limit]

    results = {}
    if os.path.exists(RESULTS) and not args.fresh:
        results = json.load(open(RESULTS, encoding="utf-8")).get("results", {})

    log_path = args.provider_log if _log_size(args.provider_log) is not None else None
    nonce = int(time.time())
    done = 0
    for item in corpus:
        iid = item["id"]
        # Resume skips finished items -- but NOT ones still contaminated by
        # provider errors, whose outcome may reflect the provider, not the engine.
        prev = results.get(iid)
        if prev and not prev.get("error") and not prev.get("provider_errors"):
            continue
        if args.kleijn:
            rec = {"format": fmt}  # the text stays in the private text folder
        else:
            rec = {"should_suggest": item["should_suggest"],
                   "phenomena": item.get("phenomena", []),
                   "must_not": item.get("must_not", []),
                   "text": item["text"]}
        attempt = 0
        while True:
            attempt += 1
            # A fresh marker per ATTEMPT: the service caches every result, the
            # degraded ones included, so a retry of the identical text was
            # answered from that cache, made no provider calls, logged no
            # errors, and was recorded as clean (found 2026-09-27; every retry
            # since the retry was added did this).
            text = item["text"] + f"\n\nTestref {nonce}-{attempt}."
            offset = _log_size(log_path)
            reported = None  # the service's own count of failed provider calls
            try:
                data = _analyze(text, fmt)
                rec["document_lint_score"] = data.get("document_lint_score")
                rec["document_level"] = data.get("document_difficulty_level")
                sugs = data.get("suggestions", {}).get("suggestions", [])
                reported = data.get("suggestions", {}).get("provider_failures")
                # drop suggestions on the cache-busting nonce block
                sugs = [s for s in sugs if "Testref" not in (s.get("original_text") or "")]
                rec["produced"] = [_slim(s) for s in sugs]
                rec["types"] = sorted({s.get("type") for s in sugs})
                rec["error"] = None
            except Exception as e:
                rec["produced"], rec["types"], rec["error"] = [], [], str(e)
            if reported is not None:
                # Exact for THIS analysis (33b57bf), and readable from any
                # machine: a run from the Mac cannot see the box's log.
                n_err = int(reported)
                by_status = {"provider_failures": n_err} if n_err else {}
                rec["validity_source"] = "response"
            else:
                # Older service without the field: scan its log, if readable.
                n_err, by_status = _provider_errors_since(log_path, offset)
                rec["validity_source"] = "log" if log_path else None
            rec["provider_errors"] = by_status
            rec["attempts"] = attempt
            if n_err == 0 or attempt > args.retries:
                break
            print(f"    {iid}: {n_err} provider error(s) {by_status} — retrying "
                  f"({attempt}/{args.retries})", flush=True)
            time.sleep(5 * attempt)
        results[iid] = rec
        done += 1
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump({"base": BASE, "nonce": nonce, "results": results}, f,
                      ensure_ascii=False, indent=1)
        print(f"[{done}] {iid}: {rec['types'] or ('ERROR: ' + rec['error'] if rec['error'] else 'none')}",
              flush=True)
        time.sleep(0.4)

    ids = [item["id"] for item in corpus]
    if args.kleijn:
        kleijn_report({i: results[i] for i in ids if i in results},
                      os.path.join(KLEIJN_DIR, "truth.json"))
        validity_report(results, ids, log_path, args.provider_log, args.retries)
        print(f"\nWrote {RESULTS} (private)")
        return

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
    validity_report(results, ids, log_path, args.provider_log, args.retries)
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
