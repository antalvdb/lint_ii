#!/usr/bin/env python3
"""Diagnostic: show which suggestion types the live server returns per sentence.

Posts a document with an in-line enumeration to /analyze, polls for the result,
and prints one line per suggestion as `sentence_index  type`, followed by a
verdict about the enumeration sentence (index 1).

Usage:
    python3 scripts/diag_enum.py
    python3 scripts/diag_enum.py https://lint-ii.valkuil.net
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "https://lint-ii.valkuil.net"

TEXT = (
    "De inspectie heeft het verbeterplan van de instelling beoordeeld. "
    "Het plan draait om het aanscherpen van de interne controles, het "
    "bijscholen van het personeel op het gebied van privacy, het vastleggen "
    "van heldere afspraken met onderaannemers en het periodiek evalueren van "
    "de gemaakte keuzes. De inspectie verwacht de eerste resultaten na een halfjaar."
)


def _get(url):
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def _post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main():
    print(f"Posting to {BASE}/analyze ...")
    job_id = _post("/analyze", {"text": TEXT})["job_id"]
    result_url = f"{BASE}/analyze-result/{job_id}"

    result = None
    for _ in range(60):
        r = _get(result_url)
        status = r.get("status")
        if status == "error":
            print("ERROR:", r.get("error"))
            return
        if status != "pending":
            result = r
            break
        time.sleep(2)
    if result is None:
        print("Timed out waiting for the analysis.")
        return

    suggestions = result["result"]["suggestions"]["suggestions"]
    print("\nsentence_index  type")
    print("--------------  ----")
    for s in sorted(suggestions, key=lambda x: x.get("sentence_index", -1)):
        print(f"{s.get('sentence_index'):>13}   {s.get('type')}")

    # Verdict for the enumeration sentence (index 1).
    on_enum_sentence = [s for s in suggestions if s.get("sentence_index") == 1]
    types = {s.get("type") for s in on_enum_sentence}
    full_rewrite = {
        "sentence_rewrite", "max_sdl", "content_words_per_clause",
        "abstract_nouns", "passive", "subordinate_clause", "sentence_length",
    }
    print("\n--- verdict (sentence 1, the enumeration) ---")
    print("types on sentence 1:", sorted(types) or "(none)")
    if "enumeration" not in types:
        print("NO enumeration suggestion was produced for sentence 1.")
    elif types & full_rewrite:
        print("BUG: an un-suppressed full-rewrite co-exists with the enumeration:",
              sorted(types & full_rewrite))
    else:
        print("OK: enumeration is the only full-sentence rewrite on sentence 1.")


if __name__ == "__main__":
    main()
