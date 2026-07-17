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
import time
import urllib.request

BASE = "https://lint-ii.valkuil.net"
HERE = os.path.dirname(__file__)

# Fields worth keeping per suggestion for judging.
KEEP = ("type", "sentence_index", "original_text", "suggested_text",
        "replacement_word", "relation", "list_intro", "list_items")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
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
    print(f"Precision={prec:.2f}  Recall={rec_:.2f}")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
