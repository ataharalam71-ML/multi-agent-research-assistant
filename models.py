"""
models.py — Shared Pydantic schemas used across all agents and the graph state.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


# ── Enums ─────────────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    PLANNER      = "planner"
    SEARCHER     = "searcher"
    WRITER       = "writer"
    CRITIC       = "critic"


class RouteDecision(str, Enum):
    SEARCH_MORE  = "search_more"
    WRITE        = "write"
    APPROVE      = "approve"
    FAIL         = "fail"


# ── Sub-models ────────────────────────────────────────────────────────────────

class SubTask(BaseModel):
    id: int
    question: str = Field(description="A focused sub-question that contributes to the main query")
    priority: int = Field(default=1, ge=1, le=3, description="1=high, 3=low")


class SearchResult(BaseModel):
    query: str
    url: str
    title: str
    content: str
    score: float = Field(default=0.0, description="Relevance score from search API")


class CritiqueResult(BaseModel):
    score: int = Field(ge=0, le=10, description="Overall quality score")
    approved: bool
    hallucination_flags: list[str] = Field(
        default_factory=list,
        description="Claims in the draft that lack source support",
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Important angles the draft missed",
    )
    feedback: str = Field(description="Concise improvement instructions for the Writer")


# ── Main graph state ───────────────────────────────────────────────────────────

class ResearchState(BaseModel):
    """
    Single source of truth threaded through every node in the LangGraph.
    LangGraph requires Annotated[list, add_messages] for the messages field.
    """
    # Input
    original_query: str = ""

    # Planner output
    sub_tasks: list[SubTask] = Field(default_factory=list)

    # Searcher output
    search_results: list[SearchResult] = Field(default_factory=list)
    search_iterations: int = 0

    # Writer output
    draft: str = ""

    # Critic output
    critique: CritiqueResult | None = None
    critic_retries: int = 0

    # Final
    final_answer: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence_score: int = 0

    # Routing
    route: RouteDecision | None = None
    error: str | None = None

    # LangGraph message thread (required for streaming)
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
