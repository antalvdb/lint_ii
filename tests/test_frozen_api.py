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
