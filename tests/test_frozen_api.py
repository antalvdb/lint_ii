"""GET /frozen/<id> and /frozen/<id>/data: stored analyses, rendered without
re-running anything (study items for human judges). No model is called."""

import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langid")
os.environ.setdefault("LINT_PROVIDER", "ollama")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

client = TestClient(api.app)
ID = "4829301746551093"


@pytest.fixture
def frozen_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_FROZEN_DIR", str(tmp_path))
    snap = {"id": ID, "model": "test-model", "result": {"document_lint_score": 41.6}}
    (tmp_path / f"{ID}.json").write_text(json.dumps(snap), encoding="utf-8")
    return tmp_path


def test_page_and_data_are_served(frozen_dir):
    page = client.get(f"/frozen/{ID}")
    assert page.status_code == 200 and "lint-ii-visualizer" in page.text
    data = client.get(f"/frozen/{ID}/data")
    assert data.status_code == 200
    assert data.json()["result"]["document_lint_score"] == 41.6


@pytest.mark.parametrize("bad", [
    "4829301746551094",          # well formed, not stored
    "482930174655109",           # 15 digits
    "48293017465510931",         # 17 digits
    "48293017465510a3",          # not all digits
])
def test_unknown_or_malformed_ids_are_404(frozen_dir, bad):
    assert client.get(f"/frozen/{bad}").status_code == 404
    assert client.get(f"/frozen/{bad}/data").status_code == 404


def test_other_json_in_the_directory_is_not_reachable(frozen_dir):
    (frozen_dir / "index.json").write_text("{}", encoding="utf-8")
    assert client.get("/frozen/index/data").status_code == 404


def test_page_assets_resolve_from_the_frozen_path():
    # /frozen/<id> is one level deeper than /, so relative asset paths would
    # 404 there: the page must reference its scripts absolutely.
    html = open(os.path.join(os.path.dirname(api.__file__), "editor_demo.html"), encoding="utf-8").read()
    assert 'src="./' not in html
    assert 'src="/src/visualizer/lint_ii_visualizer.js' in html


def _log_lines(frozen_dir):
    path = frozen_dir / "logs" / f"{ID}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_interactions_are_appended_to_the_log(frozen_dir):
    body = {"participant": "P07", "session": "0b6c-41aa", "events": [
        {"seq": 0, "t": "2026-09-30T14:00:00Z", "action": "page_open"},
        {"seq": 1, "t": "2026-09-30T14:00:05Z", "action": "click_accept", "suggestion_id": "6a4b513a"},
    ]}
    assert client.post(f"/frozen/{ID}/log", json=body).status_code == 204
    # sendBeacon posts text/plain: parsed the same way, appended after.
    beacon = json.dumps({"participant": "P07", "session": "0b6c-41aa",
                         "events": [{"seq": 2, "action": "page_close", "edited_text": "Tekst."}]})
    assert client.post(f"/frozen/{ID}/log", content=beacon,
                       headers={"Content-Type": "text/plain;charset=UTF-8"}).status_code == 204
    lines = _log_lines(frozen_dir)
    assert [e["action"] for e in lines] == ["page_open", "click_accept", "page_close"]
    assert all(e["participant"] == "P07" and e["frozen_id"] == ID and e["received"] for e in lines)
    assert lines[1]["suggestion_id"] == "6a4b513a"


def test_envelope_fields_cannot_be_spoofed_or_malformed(frozen_dir):
    body = {"participant": "a b/c\t+7", "session": "s" * 500,
            "events": [{"action": "click_ignore", "frozen_id": "1111111111111111",
                        "received": "then", "participant": "someone else"}]}
    assert client.post(f"/frozen/{ID}/log", json=body).status_code == 204
    (event,) = _log_lines(frozen_dir)
    assert event["frozen_id"] == ID and event["received"] != "then"
    # The survey's code is kept whatever its characters: only the tab goes.
    assert event["participant"] == "a b/c+7"
    assert event["session"] == "s" * 128          # clipped


@pytest.mark.parametrize("content", ["not json", json.dumps({"events": "x"}), json.dumps([1])])
def test_malformed_log_bodies_are_400(frozen_dir, content):
    assert client.post(f"/frozen/{ID}/log", content=content).status_code == 400
    assert not (frozen_dir / "logs").exists()


def test_log_for_unknown_id_is_404(frozen_dir):
    assert client.post("/frozen/4829301746551094/log", json={"events": []}).status_code == 404


def test_oversized_event_is_kept_as_a_truncation_marker(frozen_dir):
    body = {"events": [{"seq": 3, "action": "copy_result", "text": "x" * 40_000}]}
    assert client.post(f"/frozen/{ID}/log", json=body).status_code == 204
    (event,) = _log_lines(frozen_dir)
    assert event["truncated"] is True and event["action"] == "copy_result" and "text" not in event
