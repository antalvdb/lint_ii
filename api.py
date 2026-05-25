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
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.INFO)
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
    max_suggestions: int = Field(default=5, ge=1, le=10)


def _run_analysis(text: str, max_suggestions: int) -> dict:
    from lint_ii import ReadabilityAnalysis
    from lint_ii.llm.suggestions import SuggestionEngine

    t0 = time.perf_counter()
    analysis = ReadabilityAnalysis.from_text(text)
    t1 = time.perf_counter()
    logger.info("TIMING spacy_analysis=%.2fs", t1 - t0)

    engine = SuggestionEngine(provider=_provider)
    suggestions = engine.generate_suggestions(analysis, max_suggestions=max_suggestions)
    t2 = time.perf_counter()
    logger.info("TIMING llm_suggestions=%.2fs total=%.2fs", t2 - t1, t2 - t0)

    return analysis.with_suggestions(suggestions).as_dict()


@app.get("/health")
def health():
    return {"status": "ok", "model": _provider.model_name if _provider else None}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        logger.info("Starting analysis (%d chars)", len(request.text))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_analysis,
            request.text,
            request.max_suggestions,
        )
        n = len(result.get("suggestions", {}).get("suggestions", []))
        logger.info("Analysis complete: %d suggestions", n)
        return result
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Serve the demo frontend as static files (must be last)
app.mount("/", StaticFiles(directory=".", html=True), name="static")
