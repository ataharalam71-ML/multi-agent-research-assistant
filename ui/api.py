"""
ui/api.py — FastAPI REST API for the research assistant.
Run with:  uvicorn ui.api:app --reload

Endpoints:
  POST /research          → run full pipeline, returns JSON
  POST /research/stream   → SSE stream of node updates
  GET  /health            → health check
"""
from __future__ import annotations
import asyncio
import json
import logging
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph import get_graph
from models import ResearchState
from memory.store import VectorMemory
from config.settings import get_settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Research API",
    description="LangGraph-powered research assistant with 4 specialised agents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    session_id: str | None = None


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    confidence_score: int
    sources: list[str]
    critic_retries: int
    search_iterations: int


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def run_research(req: ResearchRequest):
    session_id = req.session_id or str(uuid.uuid4())
    graph = get_graph()

    try:
        initial = ResearchState(original_query=req.query)
        result  = await asyncio.to_thread(graph.invoke, initial)
        state   = ResearchState(**result)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    answer = state.final_answer or state.draft

    # Persist
    try:
        vm = VectorMemory(persist_dir=get_settings().chroma_persist_dir)
        vm.save_session(query=req.query, answer=answer, session_id=session_id)
    except Exception:
        pass

    return ResearchResponse(
        session_id       = session_id,
        query            = req.query,
        answer           = answer,
        confidence_score = state.confidence_score,
        sources          = state.sources,
        critic_retries   = state.critic_retries,
        search_iterations = state.search_iterations,
    )


@app.post("/research/stream")
async def stream_research(req: ResearchRequest):
    """
    Server-Sent Events stream. Each event is a JSON object:
    {"node": "planner"|"searcher"|"writer"|"critic", "data": {...}}
    """
    graph = get_graph()

    async def event_generator() -> AsyncIterator[str]:
        initial = ResearchState(original_query=req.query)

        def _run_stream():
            return list(graph.stream(initial, stream_mode="updates"))

        try:
            chunks = await asyncio.to_thread(_run_stream)
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        for chunk in chunks:
            for node_name, delta in chunk.items():
                payload = {
                    "node": node_name,
                    "data": {
                        k: (v.model_dump() if hasattr(v, "model_dump") else v)
                        for k, v in delta.items()
                        if k != "messages"
                    },
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)  # yield to event loop

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
