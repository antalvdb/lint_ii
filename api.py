"""
LiNT-II demo API.

Accepts Dutch text, runs readability analysis + LLM suggestion generation
via a local Ollama server, and returns the full visualization payload.

Usage:
    /Users/antalb/opt/miniconda3/envs/py311/bin/uvicorn api:app --host 0.0.0.0 --port 8080
"""

import sys
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LiNT-II Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# One worker: Ollama handles one request at a time anyway
_executor = ThreadPoolExecutor(max_workers=1)
_busy = False  # simple busy flag to inform callers


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10_000)
    model: str = Field(default="qwen2.5:14b")
    max_suggestions: int = Field(default=5, ge=1, le=10)


def _run_analysis(text: str, model: str, max_suggestions: int) -> dict:
    from lint_ii import ReadabilityAnalysis
    analysis = ReadabilityAnalysis.from_text(text)
    suggestions = analysis.generate_suggestions(
        llm_config={"provider": "ollama", "model": model},
        max_suggestions=max_suggestions,
    )
    return analysis.with_suggestions(suggestions).as_dict()


@app.get("/health")
def health():
    return {"status": "ok", "busy": _busy}


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
        logger.info("Starting analysis (%d chars, model=%s)", len(request.text), request.model)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_analysis,
            request.text,
            request.model,
            request.max_suggestions,
        )
        logger.info("Analysis complete: %d suggestions", len(result.get("suggestions", {}).get("suggestions", [])))
        return result
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _busy = False


# Serve the demo frontend as static files (must be last)
app.mount("/", StaticFiles(directory=".", html=True), name="static")
