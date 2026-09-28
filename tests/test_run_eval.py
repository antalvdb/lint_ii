"""The eval runner (scripts/eval/run_eval.py): the provider-error gate, the
Kleijn mode's items and permission gate, and retries that really re-analyse.
No server is contacted: the network calls are replaced."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "eval"))

import run_eval  # noqa: E402


def test_gate_counts_http_errors_and_unanswered_calls(tmp_path):
    log = tmp_path / "app.log"
    log.write_text(
        'INFO:httpx:HTTP Request: POST https://x/v1/chat/completions "HTTP/1.1 429 Too Many Requests"\n'
        'INFO:httpx:HTTP Request: POST https://x/v1/chat/completions "HTTP/1.1 200 OK"\n'
        "Traceback (most recent call last):\n"
        "httpcore.ConnectError: [Errno 61] Connection refused\n"
        "httpx.ConnectError: [Errno 61] Connection refused\n"
        "httpx.ReadTimeout: timed out\n"
    )
    n, by = run_eval._provider_errors_since(str(log), 0)
    assert by == {"429": 1, "ConnectError": 1, "ReadTimeout": 1}
    assert n == 3


def test_kleijn_items_filter_versions_and_keep_text_order(tmp_path):
    for name in ("T02_moei.txt", "T01_mak.txt", "T01_moei.txt", "notes.txt"):
        (tmp_path / name).write_text(f"### Titel\n\nTekst van {name}.", encoding="utf-8")
    assert [i["id"] for i in run_eval.kleijn_items(str(tmp_path), {"moei"})] == ["T01_moei", "T02_moei"]
    assert [i["id"] for i in run_eval.kleijn_items(str(tmp_path), {"mak", "moei"})] == [
        "T01_mak", "T01_moei", "T02_moei"]


def test_kleijn_mode_refuses_without_owners_ok(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--kleijn"])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert "owners" in str(e.value)


def test_a_retry_sends_a_fresh_text_not_a_cached_one(monkeypatch, tmp_path):
    """The service caches every result, degraded ones included. A retry of the
    identical text was answered from that cache and recorded as clean."""
    kdir = tmp_path / "kleijn"
    (kdir / "texts").mkdir(parents=True)
    (kdir / "texts" / "T01_moei.txt").write_text("### Titel\n\nDe trein was laat.", encoding="utf-8")
    (kdir / "truth.json").write_text(json.dumps({"texts": {}}), encoding="utf-8")
    log = tmp_path / "app.log"
    log.write_text("")
    monkeypatch.setattr(run_eval, "KLEIJN_DIR", str(kdir))
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: None)

    sent = []
    monkeypatch.setattr(run_eval, "_analyze", lambda text, fmt="text": (
        sent.append((text, fmt)) or {"suggestions": {"suggestions": []}}))
    errors = iter([(1, {"429": 1}), (0, {})])  # the first attempt hit a 429
    monkeypatch.setattr(run_eval, "_provider_errors_since", lambda path, offset: next(errors))
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--kleijn", "--owners-ok", "--retries", "2",
        "--results", str(tmp_path / "results.json"), "--provider-log", str(log)])
    run_eval.main()

    assert len(sent) == 2
    assert sent[0][0] != sent[1][0], "the retry must not resend the identical (cached) text"
    assert all(fmt == "markdown" for _, fmt in sent)
    rec = json.loads((tmp_path / "results.json").read_text())["results"]["T01_moei"]
    assert rec["attempts"] == 2 and rec["provider_errors"] == {}
    assert "text" not in rec  # the Kleijn texts never enter the results file


def test_the_services_own_count_drives_retries_without_a_log(monkeypatch, tmp_path, capsys):
    """A run from the Mac cannot read the box's log. The service reports
    suggestions.provider_failures itself (33b57bf); that alone must trigger the
    retry and make the item count as checked."""
    kdir = tmp_path / "kleijn"
    (kdir / "texts").mkdir(parents=True)
    (kdir / "texts" / "T01_moei.txt").write_text("### Titel\n\nDe trein was laat.", encoding="utf-8")
    (kdir / "truth.json").write_text(json.dumps({"texts": {}}), encoding="utf-8")
    monkeypatch.setattr(run_eval, "KLEIJN_DIR", str(kdir))
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: None)

    failures = iter([2, 0])  # the first attempt lost two provider calls
    monkeypatch.setattr(run_eval, "_analyze", lambda text, fmt="text": {
        "suggestions": {"suggestions": [], "provider_failures": next(failures)}})
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--kleijn", "--owners-ok", "--retries", "2",
        "--results", str(tmp_path / "results.json"),
        "--provider-log", str(tmp_path / "no-such.log")])
    run_eval.main()

    rec = json.loads((tmp_path / "results.json").read_text())["results"]["T01_moei"]
    assert rec["attempts"] == 2 and rec["provider_errors"] == {}
    assert rec["validity_source"] == "response"
    out = capsys.readouterr().out
    assert "retrying" in out and "VALIDITY: CLEAN" in out and "NOT CHECKED" not in out
