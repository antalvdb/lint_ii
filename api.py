"""
LiNT-II demo API.

Accepts Dutch text, runs readability analysis + LLM suggestion generation
and returns the full visualization payload.

Provider is selected via the LINT_PROVIDER env var (default: mlx).
  LINT_PROVIDER=mlx       Apple Silicon, model path via LINT_MODEL (default: mlx-community/Qwen2.5-14B-Instruct-4bit)
  LINT_PROVIDER=ollama    Ollama server, model name via LINT_MODEL (default: qwen2.5:72b)
  LINT_PROVIDER=mistral   Mistral API, model via LINT_MODEL (default: mistral-large-latest);
                          key via MISTRAL_API_KEY. NOTE: sends tester text to
                          an external service, and LINT_MODEL must be changed
                          away from the MLX model path when switching.

Usage (Mac):
    /Users/antalb/opt/miniconda3/envs/py311/bin/uvicorn api:app --host 0.0.0.0 --port 8443 ...

Usage (Linux/Ollama):
    LINT_PROVIDER=ollama LINT_MODEL=qwen2.5:72b uvicorn api:app --host 0.0.0.0 --port 8443 ...
"""

import sys
import os
import asyncio
import hashlib
import json
import logging
import logging.handlers
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Logging: the DEBUG stream (full LLM prompts/responses — the primary
# debugging tool, but also the growth driver) goes to a size-capped rotating
# file; launchd's err.log only receives INFO and up, so it stays small.
# Timestamps are included in both (the 2026-06-10 wedge diagnosis had to
# reconstruct timing from file mtimes).
_LOG_FORMAT = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
_APP_LOG_PATH = os.path.expanduser("~/Library/Logs/lint-ii.app.log")

_stderr_handler = logging.StreamHandler()
_stderr_handler.setLevel(logging.INFO)
_file_handler = logging.handlers.RotatingFileHandler(
    _APP_LOG_PATH, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
logging.basicConfig(
    level=logging.DEBUG,
    format=_LOG_FORMAT,
    handlers=[_stderr_handler, _file_handler],
)
# httpx/httpcore emit several DEBUG lines per request; useless bulk.
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)
# Heavy LLM analyses run one at a time. The MLX 32B model is a single shared
# resource on a 32 GB machine; running several /analyze jobs concurrently
# thrashes memory and wedges the server, so they queue on a single worker while
# the fast /analyze-lint and /convert paths stay on the 4-worker pool.
_analysis_executor = ThreadPoolExecutor(max_workers=1)
_provider = None

LINT_PROVIDER = os.environ.get("LINT_PROVIDER", "mlx")
LINT_MODEL = os.environ.get("LINT_MODEL", None)


def _load_provider():
    from lint_ii.llm.providers import create_provider
    kwargs = {"model": LINT_MODEL} if LINT_MODEL else {}
    provider = create_provider(LINT_PROVIDER, **kwargs)
    if hasattr(provider, "load"):
        logger.info("Loading %s model %s …", LINT_PROVIDER, provider.model_name)
        t0 = time.perf_counter()
        provider.load()
        logger.info("Model ready in %.1fs", time.perf_counter() - t0)
    else:
        logger.info("Using %s provider with model %s", LINT_PROVIDER, provider.model_name)
    return provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider
    loop = asyncio.get_event_loop()
    _provider = await loop.run_in_executor(_executor, _load_provider)
    # After the provider is known (its name is part of the cache key): restore
    # cached analyses from disk, then queue warm-up analyses for any example
    # text not yet cached.
    _cache_load()
    _warm_example_cache()
    yield


app = FastAPI(title="LiNT-II Demo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_html(request, call_next):
    """Force browsers to revalidate HTML so a cached editor_demo.html doesn't
    keep loading stale ?v= asset references (notably on iOS WebKit). Static
    JS/CSS keep their own caching — they are versioned via ?v= query strings."""
    response = await call_next(request)
    path = request.url.path
    if path.endswith(".html") or path in ("/", ""):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20_000)
    # None means "one readability suggestion per sentence" — resolved at
    # runtime from the analysed sentence count (see _run_analysis).
    max_suggestions: int | None = Field(default=None, ge=1, le=50)
    # "text" (line-heuristic structure) or "markdown" (AST structure, e.g. from
    # a converted .docx — see /convert).
    format: str = Field(default="text")


def _build_analysis(text: str, fmt: str):
    """Build a ReadabilityAnalysis, using Markdown structure when requested."""
    from lint_ii import ReadabilityAnalysis
    if fmt == "markdown":
        return ReadabilityAnalysis.from_markdown(text)
    return ReadabilityAnalysis.from_text(text)


def _run_lint_only(text: str, fmt: str = "text") -> dict:
    t0 = time.perf_counter()
    analysis = _build_analysis(text, fmt)
    logger.info("TIMING spacy_analysis=%.2fs", time.perf_counter() - t0)
    result = analysis.as_dict()
    result['suggestions'] = {'suggestions': [], 'triggers_found': 0, 'triggers_processed': 0, 'model': ''}
    return result


# Ceiling for the defaulted suggestion budget (= LLM calls). Without it a
# document near the 20k-char limit would plan ~300 jobs and occupy the
# single-worker LLM executor for tens of minutes, starving every other client.
_MAX_DEFAULT_SUGGESTIONS = 25


def _run_analysis(text: str, max_suggestions: int | None, fmt: str = "text") -> dict:
    from lint_ii.llm.suggestions import SuggestionEngine

    t0 = time.perf_counter()
    analysis = _build_analysis(text, fmt)
    t1 = time.perf_counter()
    logger.info("TIMING spacy_analysis=%.2fs", t1 - t0)

    # Default to roughly one readability suggestion per 10 words, capped at
    # _MAX_DEFAULT_SUGGESTIONS jobs (~5-7s of LLM time each). Combined with
    # the round-robin in _prioritize_triggers, this spreads length-proportional
    # coverage across sentences (at least one suggestion for any non-trivial text).
    if max_suggestions is None:
        word_count = sum(
            1 for sent in analysis.sentences
            for tok in sent.word_features
            if not tok.is_punctuation
        )
        max_suggestions = min(max(1, round(word_count / 10)), _MAX_DEFAULT_SUGGESTIONS)
        logger.info(
            "max_suggestions defaulted to %d (word_count=%d, ~1 per 10 words, cap %d)",
            max_suggestions, word_count, _MAX_DEFAULT_SUGGESTIONS,
        )

    engine = SuggestionEngine(provider=_provider)
    suggestions = engine.generate_suggestions(analysis, max_suggestions=max_suggestions)
    t2 = time.perf_counter()
    logger.info("TIMING llm_suggestions=%.2fs total=%.2fs", t2 - t1, t2 - t0)

    return analysis.with_suggestions(suggestions).as_dict()


def _pending_job_stats() -> dict:
    """Pending-job count and the age of the oldest one, for /health.

    A pending job's ts is its creation time, so the age covers the full
    client-visible wait (queue time + LLM run). A slowly climbing age is
    normal under a queue of long documents; an age far beyond the watchdog
    timeout means jobs are not being drained and warrants a look."""
    now = time.time()
    with _jobs_lock:
        ages = [now - j["ts"] for j in _jobs.values() if j["status"] == "pending"]
    stats = {"pending_jobs": len(ages)}
    if ages:
        stats["oldest_pending_seconds"] = round(max(ages))
    return stats


@app.get("/health")
def health():
    from fastapi.responses import JSONResponse
    from lint_ii.llm.providers import llm_wedged_since

    wedged_since = llm_wedged_since()
    if wedged_since is not None:
        # An LLM call blew through its watchdog timeout and the Metal driver is
        # presumed wedged; jobs fail fast until the server is restarted. 503 so
        # external monitoring finally sees the incident (the 2026-06-10 wedge
        # kept /health green all day).
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": "llm-watchdog-timeout",
                "wedged_for_seconds": round(time.time() - wedged_since),
                "model": _provider.model_name if _provider else None,
                **_pending_job_stats(),
            },
        )
    return {
        "status": "ok",
        "model": _provider.model_name if _provider else None,
        **_pending_job_stats(),
    }


@app.post("/analyze-lint")
async def analyze_lint(request: AnalyzeRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _run_lint_only, request.text, request.format
        )
        return result
    except Exception as e:
        logger.error("Lint analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Job store for the async analyze flow. A single long /analyze request was
# aborted by iOS WebKit mid-run (idle connection), so the client kicks off a
# job and then polls a fast status endpoint instead — no long-lived request.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_JOB_MAX_AGE = 600  # seconds; prune unclaimed finished jobs after this

# LRU cache of completed analyses keyed by request content. Testers mostly
# re-run the same four example texts; serving a repeat from cache makes it
# near-instant and keeps the LLM worker free for new documents. Suggestions
# vary run-to-run anyway (sampling), so pinning repeats to one result is fine.
# The cache is written through to disk so restarts do not empty it, and the
# example texts are warmed at startup (see _warm_example_cache).
_RESULT_CACHE_MAX = 16
_result_cache: OrderedDict[str, dict] = OrderedDict()
_result_cache_lock = threading.Lock()
_RESULT_CACHE_PATH = os.path.expanduser("~/.cache/lint-ii/result_cache.json")


def _cache_key(text: str, max_suggestions: int | None, fmt: str) -> str:
    # The model name is part of the key: the cache survives restarts on disk,
    # and a model swap must not serve results generated by the previous model.
    model = _provider.model_name if _provider else ""
    raw = f"{model}|{fmt}|{max_suggestions}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> dict | None:
    with _result_cache_lock:
        result = _result_cache.get(key)
        if result is not None:
            _result_cache.move_to_end(key)
        return result


def _cache_put(key: str, result: dict) -> None:
    with _result_cache_lock:
        _result_cache[key] = result
        _result_cache.move_to_end(key)
        while len(_result_cache) > _RESULT_CACHE_MAX:
            _result_cache.popitem(last=False)
        snapshot = list(_result_cache.items())
    _cache_save(snapshot)


def _cache_save(snapshot: list[tuple[str, dict]]) -> None:
    """Write the cache to disk (atomically, via a temp file). Runs outside the
    cache lock; a failed write only costs persistence, never correctness."""
    try:
        os.makedirs(os.path.dirname(_RESULT_CACHE_PATH), exist_ok=True)
        tmp_path = _RESULT_CACHE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"entries": [[k, v] for k, v in snapshot]}, f)
        os.replace(tmp_path, _RESULT_CACHE_PATH)
    except Exception as e:
        logger.warning("Could not persist result cache: %s", e)


def _cache_load() -> None:
    try:
        with open(_RESULT_CACHE_PATH, encoding="utf-8") as f:
            entries = json.load(f)["entries"]
    except FileNotFoundError:
        return
    except Exception as e:
        logger.warning("Could not load persisted result cache: %s", e)
        return
    with _result_cache_lock:
        for key, result in entries[-_RESULT_CACHE_MAX:]:
            _result_cache[key] = result
    logger.info("Loaded %d cached analyses from disk", len(entries))


def _load_example_texts() -> dict[str, str]:
    """Extract the EXAMPLE_TEXTS template literals from editor_demo.html so the
    demo page stays the single source of truth for the example texts."""
    import re
    path = os.path.join(os.path.dirname(__file__), "editor_demo.html")
    try:
        with open(path, encoding="utf-8") as f:
            html = f.read()
        block = re.search(r"const EXAMPLE_TEXTS = \{(.*?)\n\s*\}", html, re.DOTALL)
        pairs = re.findall(r"(\d+):\s*`([^`]+)`", block.group(1))
        return {name: text for name, text in pairs}
    except Exception as e:
        logger.warning("Could not extract example texts from %s: %s", path, e)
        return {}


def _warm_example_cache() -> None:
    """Queue an analysis for every example text not already cached, so the
    example buttons are instant from the first click after a (re)start. Warm
    jobs share the single-worker LLM queue, so tester requests submitted
    meanwhile simply interleave FIFO."""
    examples = _load_example_texts()
    if not examples:
        return
    warmed = 0
    for name, text in sorted(examples.items()):
        key = _cache_key(text, None, "text")
        if _cache_get(key) is not None:
            continue
        warmed += 1

        def _job(name=name, text=text, key=key):
            try:
                t0 = time.perf_counter()
                result = _run_analysis(text, None, "text")
                _cache_put(key, result)
                logger.info(
                    "Warmed example %s into result cache (%.1fs)",
                    name, time.perf_counter() - t0,
                )
            except Exception as e:
                logger.warning("Cache warm for example %s failed: %s", name, e)

        _analysis_executor.submit(_job)
    logger.info(
        "Example cache warm: %d of %d examples queued (%d already cached)",
        warmed, len(examples), len(examples) - warmed,
    )


def _store_job_result(job_id: str, cache_key: str, fut) -> None:
    if fut.cancelled():
        logger.info("Analysis job %s cancelled while queued", job_id)
        with _jobs_lock:
            _jobs.pop(job_id, None)
        return
    try:
        result = fut.result()
        n = len(result.get("suggestions", {}).get("suggestions", []))
        logger.info("Analysis job %s complete: %d suggestions", job_id, n)
        # Cache even if the requesting client cancelled meanwhile — the work is
        # done, so the next identical request might as well benefit from it.
        _cache_put(cache_key, result)
        entry = {"status": "done", "result": result, "ts": time.time()}
    except Exception as e:
        logger.error("Analysis job %s failed: %s", job_id, e, exc_info=True)
        from lint_ii.llm.providers import LLMTimeoutError
        if isinstance(e, LLMTimeoutError):
            msg = (
                "Het taalmodel reageert niet; de analyse is afgebroken. "
                "De demo-server moet waarschijnlijk opnieuw worden gestart — "
                "probeer het later nog eens."
            )
        else:
            msg = str(e)
        entry = {"status": "error", "error": msg, "ts": time.time()}
    with _jobs_lock:
        if job_id not in _jobs:
            # Cancelled while running: nobody is polling anymore, drop the result.
            logger.info("Analysis job %s finished after cancellation; result discarded", job_id)
            return
        _jobs[job_id] = entry


def _prune_jobs() -> None:
    """Drop finished jobs whose result was never picked up. Pending jobs are
    never pruned: they may legitimately wait in the single-worker queue longer
    than _JOB_MAX_AGE, and pruning them would 404 a client still polling."""
    now = time.time()
    with _jobs_lock:
        stale = [
            jid for jid, j in _jobs.items()
            if j["status"] != "pending" and now - j.get("ts", now) > _JOB_MAX_AGE
        ]
        for jid in stale:
            _jobs.pop(jid, None)


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Start an analysis job and return its id immediately (poll /analyze-result)."""
    _prune_jobs()
    job_id = uuid.uuid4().hex[:12]
    cache_key = _cache_key(request.text, request.max_suggestions, request.format)
    cached = _cache_get(cache_key)
    if cached is not None:
        # Known text (e.g. an example button): store the job as already done so
        # the client's normal poll picks it up immediately — no LLM run at all.
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "result": cached, "ts": time.time()}
        logger.info("Analysis job %s served from cache (%d chars)", job_id, len(request.text))
        return {"job_id": job_id}
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "ts": time.time()}
    fut = _analysis_executor.submit(_run_analysis, request.text, request.max_suggestions, request.format)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None and job["status"] == "pending":
            # Keep the Future so /analyze-cancel can dequeue the job before it runs.
            job["future"] = fut
    fut.add_done_callback(lambda f: _store_job_result(job_id, cache_key, f))
    logger.info("Analysis job %s started (%d chars, format=%s)", job_id, len(request.text), request.format)
    return {"job_id": job_id}


@app.post("/analyze-cancel/{job_id}")
async def analyze_cancel(job_id: str):
    """Cancel an analysis job (sent on re-analyse or page close).

    A job still waiting in the queue is dequeued outright. A job already
    running cannot be interrupted mid-LLM-call, but its entry is dropped so
    the result is discarded when it finishes."""
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job is None:
        return {"cancelled": False}
    fut = job.get("future")
    dequeued = bool(fut is not None and fut.cancel())
    logger.info(
        "Analysis job %s cancelled by client (%s)",
        job_id, "dequeued" if dequeued else "already running or finished",
    )
    return {"cancelled": True, "dequeued": dequeued}


@app.get("/analyze-result/{job_id}")
async def analyze_result(job_id: str):
    """Return the status of an analysis job; delivers the result once when done."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None and job["status"] == "pending":
            # Queue position: pending jobs ahead of this one. Insertion order
            # is submission order, and the single analysis worker runs jobs
            # FIFO, so position 0 means "being analysed now".
            position = 0
            for jid, j in _jobs.items():
                if jid == job_id:
                    break
                if j["status"] == "pending":
                    position += 1
            return {"status": "pending", "queue_position": position}
    if job is None:
        raise HTTPException(status_code=404, detail="Onbekende of verlopen analyse.")
    # Terminal state — hand it over once and drop it from the store.
    with _jobs_lock:
        _jobs.pop(job_id, None)
    if job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    return {"status": "done", "result": job["result"]}


# --- Document upload -> Markdown (pandoc) -----------------------------------
# The client POSTs the raw file bytes as the request body with ?filename=<name>
# (no multipart dependency). pandoc converts to Markdown, which /analyze then
# segments via the AST (ReadabilityAnalysis.from_markdown), preserving the
# document's real structure (headings, lists, quotes).
def _find_pandoc() -> str | None:
    """Locate pandoc without relying on PATH (launchd starts with a minimal
    PATH that omits /opt/homebrew/bin)."""
    found = shutil.which("pandoc")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/pandoc", "/usr/local/bin/pandoc", "/usr/bin/pandoc"):
        if os.path.exists(candidate):
            return candidate
    return None


def _find_pdftotext() -> str | None:
    """Locate pdftotext (poppler) without relying on PATH."""
    found = shutil.which("pdftotext")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/pdftotext", "/usr/local/bin/pdftotext", "/usr/bin/pdftotext"):
        if os.path.exists(candidate):
            return candidate
    return None


_PANDOC = _find_pandoc()
_PDFTOTEXT = _find_pdftotext()
_UPLOAD_FORMATS = {
    ".docx": "docx", ".odt": "odt", ".rtf": "rtf",
    ".html": "html", ".htm": "html", ".epub": "epub",
    ".md": "markdown", ".markdown": "markdown", ".txt": "markdown",
}
_MAX_UPLOAD = 8 * 1024 * 1024  # 8 MB
# Mirror of AnalyzeRequest's max_length: converted text that /analyze would
# reject anyway should fail here, with a friendlier message.
_MAX_ANALYZE_CHARS = 20_000


@app.post("/convert")
async def convert(request: Request, filename: str = ""):
    """Convert an uploaded document to Markdown via pandoc.

    Returns {"markdown", "format"}; the client feeds the markdown back to
    /analyze with format="markdown".
    """
    ext = os.path.splitext(filename)[1].lower()
    is_pdf = ext == ".pdf"
    src_format = _UPLOAD_FORMATS.get(ext)
    if is_pdf:
        if _PDFTOTEXT is None:
            raise HTTPException(status_code=503, detail="PDF-conversie is niet beschikbaar op de server.")
    elif src_format is None:
        supported = ", ".join(sorted(_UPLOAD_FORMATS) + ([".pdf"] if _PDFTOTEXT else []))
        raise HTTPException(status_code=415, detail=f"Niet-ondersteund bestandstype. Ondersteund: {supported}.")
    elif _PANDOC is None:
        raise HTTPException(status_code=503, detail="Documentconversie is niet beschikbaar op de server.")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Leeg bestand ontvangen.")
    if len(data) > _MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="Bestand te groot (max 8 MB).")

    def _convert() -> str:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            if is_pdf:
                # Plain-text extraction in reading order. The output is
                # hard-wrapped at PDF line width; /analyze with format="text"
                # rejoins it into paragraphs (_unwrap_hard_breaks).
                proc = subprocess.run(
                    [_PDFTOTEXT, "-enc", "UTF-8", "-nopgbrk", tmp_path, "-"],
                    capture_output=True, timeout=30,
                )
            else:
                proc = subprocess.run(
                    [_PANDOC, "--sandbox", tmp_path, "-f", src_format, "-t", "gfm", "--wrap=none"],
                    capture_output=True, timeout=30,
                )
        finally:
            os.unlink(tmp_path)
        if proc.returncode != 0:
            tool = "pdftotext-fout" if is_pdf else "pandoc-fout"
            raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300] or tool)
        return proc.stdout.decode("utf-8", "replace")

    try:
        loop = asyncio.get_event_loop()
        converted = (await loop.run_in_executor(_executor, _convert)).strip()
    except Exception as e:
        logger.error("Conversion failed (%s): %s", filename, e)
        raise HTTPException(status_code=422, detail=f"Conversie mislukt: {e}")

    if not converted:
        raise HTTPException(status_code=422, detail="Geen tekst gevonden in het document.")
    if len(converted) > _MAX_ANALYZE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Het document bevat {len(converted):,} tekens; de demo analyseert "
                f"maximaal {_MAX_ANALYZE_CHARS:,} tekens. Kort het document in of "
                "plak een gedeelte in het tekstvak."
            ).replace(",", "."),
        )
    out_format = "text" if is_pdf else "markdown"
    logger.info("Converted %s (%s) -> %d chars %s", filename, "pdf" if is_pdf else src_format, len(converted), out_format)
    return {"markdown": converted, "format": out_format}


# Serve the demo frontend as static files (must be last)
app.mount("/", StaticFiles(directory=".", html=True), name="static")
