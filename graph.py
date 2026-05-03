"""
graph.py — Builds and compiles the LangGraph StateGraph.

Flow:
  START → planner → searcher → writer → critic
                                    ↑         |
                                    └─────────┘ (if not approved)
                                              ↓
                                           END
"""
from __future__ import annotations
import logging
from langgraph.graph import StateGraph, START, END

from agents import planner_node, searcher_node, writer_node, critic_node
from models import ResearchState, RouteDecision

logger = logging.getLogger(__name__)


# ── Routing function ───────────────────────────────────────────────────────────

def route_after_critic(state: ResearchState) -> str:
    """
    Called after the Critic node to decide the next step.
    Returns a node name (string) — LangGraph uses this for conditional edges.
    """
    if state.route == RouteDecision.APPROVE:
        logger.info("[Router] Approved → END")
        return END

    if state.route == RouteDecision.SEARCH_MORE:
        logger.info("[Router] Need more search → searcher")
        return "searcher"

    # Default: re-run writer with critique feedback
    logger.info("[Router] Need rewrite → writer")
    return "writer"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the research graph.
    Returns a CompiledGraph ready for .invoke() or .stream().
    """
    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("planner",  planner_node)
    builder.add_node("searcher", searcher_node)
    builder.add_node("writer",   writer_node)
    builder.add_node("critic",   critic_node)

    # Static edges
    builder.add_edge(START,      "planner")
    builder.add_edge("planner",  "searcher")
    builder.add_edge("searcher", "writer")
    builder.add_edge("writer",   "critic")

    # Conditional edge from critic
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            END:       END,
            "searcher": "searcher",
            "writer":   "writer",
        },
    )

    return builder.compile()


# Singleton graph — compiled once and reused
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
