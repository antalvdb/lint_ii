#!/usr/bin/env python3
"""Freeze demo analyses: run each text through the live /analyze and store the
result as a snapshot that /frozen/<id> renders without re-running anything.

Run it ON THE BOX, where the service's snapshot directory lives:

    python3 scripts/freeze_analysis.py texts/*.txt
    python3 scripts/freeze_analysis.py --base http://127.0.0.1:8000 A.txt B.txt

Each text is analysed as if it were pasted into the demo (format "text"; lines
starting with "## " lose that marker first, since the demo's text mode detects
headings from the line layout). A result with failed provider calls is
incomplete and is NOT frozen unless --allow-incomplete is given.

Every snapshot gets a new random 16-digit id: a frozen analysis never changes,
so freezing a text again gives a new URL. The script prints one line per text
(file, URL, LiNT score, level, number of suggestions) and appends the same to
index.tsv in the snapshot directory, so no URL is lost.
"""

import argparse
import datetime
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.request

ID_LEN = 16
PUBLIC_BASE = "https://lint-ii.valkuil.net"


def _request(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def analyze(base: str, text: str, max_wait: float) -> dict:
    job = _request(f"{base}/analyze", {"text": text, "format": "text"})
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        poll = _request(f"{base}/analyze-result/{job['job_id']}")
        if poll.get("status") == "done" or "result" in poll:
            return poll["result"]
        if poll.get("status") == "error":
            raise RuntimeError(poll.get("error") or "analysis failed")
        time.sleep(3)
    raise TimeoutError(f"no result after {max_wait:.0f} s")


def new_id(directory: str) -> str:
    while True:
        frozen_id = str(secrets.randbelow(9 * 10 ** (ID_LEN - 1)) + 10 ** (ID_LEN - 1))
        if not os.path.exists(os.path.join(directory, f"{frozen_id}.json")):
            return frozen_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="service to analyse with")
    ap.add_argument("--public", default=PUBLIC_BASE, help="base of the printed URLs")
    ap.add_argument("--dir", default=os.environ.get("LINT_II_FROZEN_DIR")
                    or os.path.expanduser("~/.local/share/lint-ii/frozen"),
                    help="snapshot directory; must be the one the service reads")
    ap.add_argument("--max-wait", type=float, default=600)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="freeze even when some provider calls failed")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    health = _request(f"{args.base}/health")
    if health.get("restart_needed"):
        print("Refusing: the service runs older code than is checked out "
              "(/health restart_needed). Restart it first.", file=sys.stderr)
        return 2

    failures = 0
    for path in args.files:
        text = re.sub(r"(?m)^## ", "", open(path, encoding="utf-8").read()).strip()
        try:
            result = analyze(args.base, text, args.max_wait)
        except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
            print(f"{path}\tFAILED: {e}", file=sys.stderr)
            failures += 1
            continue
        sugg = result.get("suggestions") or {}
        if sugg.get("provider_failures") and not args.allow_incomplete:
            print(f"{path}\tNOT FROZEN: {sugg['provider_failures']} provider call(s) failed; "
                  "run again, or pass --allow-incomplete", file=sys.stderr)
            failures += 1
            continue
        frozen_id = new_id(args.dir)
        snapshot = {
            "id": frozen_id,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "source_file": os.path.basename(path),
            "commit": health.get("commit"),
            "model": sugg.get("model") or health.get("model"),
            "text": text,
            "format": "text",
            "result": result,
        }
        tmp = os.path.join(args.dir, f".{frozen_id}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(args.dir, f"{frozen_id}.json"))
        url = f"{args.public}/frozen/{frozen_id}"
        n = len(sugg.get("suggestions") or [])
        line = (f"{os.path.basename(path)}\t{url}\t{result.get('document_lint_score', 0):.1f}\t"
                f"{result.get('document_difficulty_level')}\t{n}\t{snapshot['created']}\t{snapshot['commit']}")
        print(line)
        with open(os.path.join(args.dir, "index.tsv"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
