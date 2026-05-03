"""
agents/critic.py — Evaluates the Writer's draft for accuracy, completeness,
and hallucinations. Returns a structured CritiqueResult and routing decision.
"""
from __future__ import annotations
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from config.llm import get_llm
from config.settings import get_settings
from models import ResearchState, CritiqueResult, RouteDecision

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a rigorous fact-checker and quality critic.
Evaluate the draft answer against the provided sources.

Check for:
1. Hallucinations — claims not supported by any listed source
2. Missing aspects — important angles the query required but the draft ignored
3. Citation accuracy — does [1] actually support the sentence that cites it?
4. Clarity and completeness

Scoring:
- 9-10: Excellent, approve immediately
- 7-8:  Good, minor issues, approve
- 5-6:  Needs improvement, request rewrite
- 0-4:  Major issues, request more search + rewrite

Return ONLY valid JSON — no markdown, no extra text:
{
  "score": <0-10>,
  "approved": <true|false>,
  "hallucination_flags": ["claim that lacks support", ...],
  "missing_aspects": ["topic not covered", ...],
  "feedback": "Concise instructions for the Writer to improve the draft."
}

approved = true if score >= 7, false otherwise.
"""


def _format_for_critic(state: ResearchState) -> str:
    sources_text = "\n".join(
        f"[{i}] {r.title}: {r.content[:200]}"
        for i, r in enumerate(state.search_results, start=1)
    )
    return f"""
Query: {state.original_query}

Draft:
{state.draft}

Sources available to the writer:
{sources_text}
"""


def critic_node(state: ResearchState) -> dict:
    """
    LangGraph node. Reads state.draft + state.search_results.
    Writes state.critique, state.route, and potentially state.final_answer.
    """
    settings = get_settings()
    logger.info("[Critic] Evaluating draft (retry %d/%d)",
                state.critic_retries, settings.max_critic_retries)

    # Hard stop: exceeded max retries → accept the best we have
    if state.critic_retries >= settings.max_critic_retries:
        logger.warning("[Critic] Max retries reached — approving current draft")
        critique = CritiqueResult(
            score=6,
            approved=True,
            feedback="Max retries reached; accepting current draft.",
        )
        return {
            "critique":        critique,
            "final_answer":    state.draft,
            "confidence_score": critique.score,
            "route":           RouteDecision.APPROVE,
        }

    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_format_for_critic(state)),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content
        if isinstance(raw_text, list):
             raw_text = " ".join(
                  block.get("text", "") if isinstance(block, dict) else str(block)
                  for block in raw_text
 )
        raw_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()

        parser = JsonOutputParser()
        data = parser.parse(raw_text)
        critique = CritiqueResult(**data)

    except (ValidationError, Exception) as exc:
        logger.error("[Critic] Parse failed: %s", exc)
        # Fallback: approve with low score to avoid infinite loops
        critique = CritiqueResult(
            score=5,
            approved=True,
            feedback=f"Critic parse error ({exc}); proceeding with current draft.",
        )

    logger.info("[Critic] Score=%d approved=%s", critique.score, critique.approved)

    if critique.approved:
        route = RouteDecision.APPROVE
        final_answer = state.draft
    elif state.critique and not state.critique.approved:
        # Second failure → try getting more search data
        route = RouteDecision.SEARCH_MORE
        final_answer = ""
    else:
        route = RouteDecision.WRITE
        final_answer = ""

    return {
        "critique":         critique,
        "critic_retries":   state.critic_retries + 1,
        "route":            route,
        "final_answer":     final_answer,
        "confidence_score": critique.score,
    }
