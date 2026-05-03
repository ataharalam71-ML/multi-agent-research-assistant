"""
agents/writer.py — Synthesises all search results into a well-structured,
cited answer in Markdown format.
"""
from __future__ import annotations
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from config.llm import get_llm
from models import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert research writer. Your job is to write a comprehensive,
well-structured answer based on the provided sources.

Writing rules:
1. Use Markdown: H2 headers for sections, bullet points for lists
2. Cite sources inline as [1], [2], etc.
3. Be factual — only include claims supported by the provided sources
4. End with a "## Sources" section listing all cited URLs
5. Aim for 300–600 words unless the topic demands more
6. If sources contradict each other, note both perspectives

Format your answer as:
## [Topic]

[Introduction paragraph]

## [Section 1]
...

## [Section 2]
...

## Sources
[1] Title — URL
[2] Title — URL
"""


def _format_sources(state: ResearchState) -> str:
    lines = []
    for i, r in enumerate(state.search_results, start=1):
        snippet = r.content[:300].replace("\n", " ")
        lines.append(f"[{i}] {r.title}\nURL: {r.url}\nSnippet: {snippet}\n")
    return "\n".join(lines)


def writer_node(state: ResearchState) -> dict:
    """
    LangGraph node. Reads state.search_results (and optionally state.critique),
    writes state.draft and state.sources.
    """
    logger.info("[Writer] Drafting answer from %d sources", len(state.search_results))

    llm = get_llm(temperature=0.3)  # slight creativity for prose quality

    critique_section = ""
    if state.critique and not state.critique.approved:
        critique_section = f"""
Previous draft was rejected by the Critic.
Feedback to address:
{state.critique.feedback}

Hallucination flags to fix: {', '.join(state.critique.hallucination_flags) or 'none'}
Missing aspects to add: {', '.join(state.critique.missing_aspects) or 'none'}
"""

    user_prompt = f"""
Original query: {state.original_query}

{critique_section}

Available sources:
{_format_sources(state)}

Write a comprehensive answer now.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        draft = response.content
        # Handle Gemini returning a list of content blocks instead of plain string
        draft = response.content
        if isinstance(draft, list):
            draft = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in draft
            )

        # Extract cited URLs for the sources field
        cited_urls = [
            r.url for i, r in enumerate(state.search_results, start=1)
            if f"[{i}]" in draft and r.url
        ]

        logger.info("[Writer] Draft complete (%d chars, %d sources cited)",
                    len(draft), len(cited_urls))
        return {"draft": draft, "sources": cited_urls}

    except Exception as exc:
        logger.error("[Writer] Failed: %s", exc)
        return {
            "draft": f"Error generating answer: {exc}",
            "sources": [],
            "error": str(exc),
        }
