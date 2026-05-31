"""
LiNT-II demo API.

Accepts Dutch text, runs readability analysis + LLM suggestion generation
and returns the full visualization payload.

Provider is selected via the LINT_PROVIDER env var (default: mlx).
  LINT_PROVIDER=mlx       Apple Silicon, model path via LINT_MODEL (default: mlx-community/Qwen2.5-14B-Instruct-4bit)
  LINT_PROVIDER=ollama    Ollama server, model name via LINT_MODEL (default: qwen2.5:72b)

Usage (Mac):
    /Users/antalb/opt/miniconda3/envs/py311/bin/uvicorn api:app --host 0.0.0.0 --port 8443 ...

Usage (Linux/Ollama):
    LINT_PROVIDER=ollama LINT_MODEL=qwen2.5:72b uvicorn api:app --host 0.0.0.0 --port 8443 ...
"""

import sys
import os
import json
import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)
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
    yield


app = FastAPI(title="LiNT-II Demo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10_000)
    # None means "one readability suggestion per sentence" — resolved at
    # runtime from the analysed sentence count (see _run_analysis).
    max_suggestions: int | None = Field(default=None, ge=1, le=50)


def _run_lint_only(text: str) -> dict:
    from lint_ii import ReadabilityAnalysis
    t0 = time.perf_counter()
    analysis = ReadabilityAnalysis.from_text(text)
    logger.info("TIMING spacy_analysis=%.2fs", time.perf_counter() - t0)
    result = analysis.as_dict()
    result['suggestions'] = {'suggestions': [], 'triggers_found': 0, 'triggers_processed': 0, 'model': ''}
    return result


def _run_analysis(text: str, max_suggestions: int | None) -> dict:
    from lint_ii import ReadabilityAnalysis
    from lint_ii.llm.suggestions import SuggestionEngine

    t0 = time.perf_counter()
    analysis = ReadabilityAnalysis.from_text(text)
    t1 = time.perf_counter()
    logger.info("TIMING spacy_analysis=%.2fs", t1 - t0)

    # Default to roughly one readability suggestion per 10 words. Combined with
    # the round-robin in _prioritize_triggers, this spreads length-proportional
    # coverage across sentences (at least one suggestion for any non-trivial text).
    if max_suggestions is None:
        word_count = sum(
            1 for sent in analysis.sentences
            for tok in sent.word_features
            if not tok.is_punctuation
        )
        max_suggestions = max(1, round(word_count / 10))
        logger.info(
            "max_suggestions defaulted to %d (word_count=%d, ~1 per 10 words)",
            max_suggestions, word_count,
        )

    engine = SuggestionEngine(provider=_provider)
    suggestions = engine.generate_suggestions(analysis, max_suggestions=max_suggestions)
    t2 = time.perf_counter()
    logger.info("TIMING llm_suggestions=%.2fs total=%.2fs", t2 - t1, t2 - t0)

    return analysis.with_suggestions(suggestions).as_dict()


@app.get("/health")
def health():
    return {"status": "ok", "model": _provider.model_name if _provider else None}


@app.post("/analyze-lint")
async def analyze_lint(request: AnalyzeRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _run_lint_only, request.text)
        return result
    except Exception as e:
        logger.error("Lint analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Job store for the async analyze flow. A single long /analyze request was
# aborted by iOS WebKit mid-run (idle connection), so the client kicks off a
# job and then polls a fast status endpoint instead — no long-lived request.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_JOB_MAX_AGE = 600  # seconds; prune finished/stale jobs after this


def _store_job_result(job_id: str, fut) -> None:
    try:
        result = fut.result()
        n = len(result.get("suggestions", {}).get("suggestions", []))
        logger.info("Analysis job %s complete: %d suggestions", job_id, n)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "result": result, "ts": time.time()}
    except Exception as e:
        logger.error("Analysis job %s failed: %s", job_id, e, exc_info=True)
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e), "ts": time.time()}


def _prune_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if now - j.get("ts", now) > _JOB_MAX_AGE]
        for jid in stale:
            _jobs.pop(jid, None)


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Start an analysis job and return its id immediately (poll /analyze-result)."""
    _prune_jobs()
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "ts": time.time()}
    fut = _executor.submit(_run_analysis, request.text, request.max_suggestions)
    fut.add_done_callback(lambda f: _store_job_result(job_id, f))
    logger.info("Analysis job %s started (%d chars)", job_id, len(request.text))
    return {"job_id": job_id}


@app.get("/analyze-result/{job_id}")
async def analyze_result(job_id: str):
    """Return the status of an analysis job; delivers the result once when done."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Onbekende of verlopen analyse.")
    if job["status"] == "pending":
        return {"status": "pending"}
    # Terminal state — hand it over once and drop it from the store.
    with _jobs_lock:
        _jobs.pop(job_id, None)
    if job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    return {"status": "done", "result": job["result"]}


@app.post("/analyze-stream")
async def analyze_stream(request: AnalyzeRequest):
    loop = asyncio.get_event_loop()

    async def event_stream():
        try:
            from lint_ii import ReadabilityAnalysis
            from lint_ii.llm.suggestions import SuggestionEngine

            t0 = time.perf_counter()
            analysis = await loop.run_in_executor(
                _executor, lambda: ReadabilityAnalysis.from_text(request.text)
            )
            logger.info("TIMING spacy_analysis=%.2fs", time.perf_counter() - t0)

            # Send initial render payload immediately (empty suggestions)
            base = analysis.as_dict()
            base['suggestions'] = {'suggestions': [], 'triggers_found': 0, 'triggers_processed': 0, 'model': ''}
            yield f"data: {json.dumps({'type': 'init', 'data': base})}\n\n"

            engine = SuggestionEngine(provider=_provider)

            # Spelling pass
            t1 = time.perf_counter()
            spelling = await loop.run_in_executor(
                _executor, lambda: engine.generate_spelling_suggestions(analysis, _provider)
            )
            logger.info("TIMING spelling_pass=%.2fs (%d)", time.perf_counter() - t1, len(spelling))
            for s in spelling:
                yield f"data: {json.dumps({'type': 'suggestion', 'data': s.as_dict()})}\n\n"

            # Trigger passes
            triggers = engine.identify_triggers(analysis)
            triggers_to_process = engine._prioritize_triggers(triggers, request.max_suggestions)
            document_level = getattr(analysis.lint, "level", None)

            for trigger in triggers_to_process:
                t_t = time.perf_counter()
                suggestion = await loop.run_in_executor(
                    _executor,
                    lambda t=trigger: engine._generate_suggestion_for_trigger(t, _provider, document_level)
                )
                logger.info("TIMING trigger_%s=%.2fs", trigger.type.value, time.perf_counter() - t_t)
                if suggestion:
                    yield f"data: {json.dumps({'type': 'suggestion', 'data': suggestion.as_dict()})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'triggers_found': len(triggers), 'triggers_processed': len(triggers_to_process), 'model': _provider.model_name})}\n\n"

        except Exception as e:
            logger.error("Stream error: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the demo frontend as static files (must be last)
app.mount("/", StaticFiles(directory=".", html=True), name="static")
