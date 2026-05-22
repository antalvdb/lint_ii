"""
LiNT-II demo API.

Accepts Dutch text, runs readability analysis + LLM suggestion generation
using MLX on Apple Silicon (model loaded once at startup), and returns
the full visualization payload.

Usage:
    /Users/antalb/opt/miniconda3/envs/py311/bin/uvicorn api:app --host 0.0.0.0 --port 8080
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

_executor = ThreadPoolExecutor(max_workers=1)
_busy = False
_provider = None  # MLXProvider instance, loaded at startup


def _load_provider():
    from lint_ii.llm.providers import MLXProvider
    provider = MLXProvider()
    logger.info("Loading MLX model %s …", provider.model_name)
    t0 = time.perf_counter()
    provider.load()
    logger.info("MLX model ready in %.1fs", time.perf_counter() - t0)
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
    return {"status": "ok", "busy": _busy, "model": _provider.model_name if _provider else None}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    global _busy
    if _busy:
        raise HTTPException(
            status_code=503,
            detail="Een andere analyse is bezig. Probeer het over een minuut opnieuw.",
        )
    _busy = True
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
    finally:
        _busy = False


# Serve the demo frontend as static files (must be last)
app.mount("/", StaticFiles(directory=".", html=True), name="static")
