"""
agents/planner.py — Decomposes the user's query into focused sub-questions.
This is the first node in the LangGraph pipeline.
"""
from __future__ import annotations
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from config.llm import get_llm
from models import ResearchState, SubTask

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research planner. Your job is to decompose a complex query into 
2–4 focused sub-questions that together give a complete answer.

Rules:
- Each sub-question must be specific and independently answerable via web search
- Cover different angles: facts, context, recency, opposing views
- Prioritise: 1=must-answer, 2=should-answer, 3=nice-to-have
- Return ONLY valid JSON — no markdown fences, no extra text

Output format:
[
  {"id": 1, "question": "...", "priority": 1},
  {"id": 2, "question": "...", "priority": 2}
]
"""


def planner_node(state: ResearchState) -> dict:
    """
    LangGraph node. Reads state.original_query, writes state.sub_tasks.
    """
    logger.info("[Planner] Decomposing: %s", state.original_query)

    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {state.original_query}"),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content

        # Strip markdown fences if the model adds them despite instructions
        # Handle Gemini returning a list of content blocks instead of plain text
        if isinstance(raw_text, list):
            raw_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_text
)
        cleaned = raw_text.strip().removeprefix("```json").removesuffix("```").strip()

        parser = JsonOutputParser()
        raw_tasks: list[dict] = parser.parse(cleaned)

        sub_tasks = []
        for item in raw_tasks:
            try:
                sub_tasks.append(SubTask(**item))
            except (ValidationError, TypeError) as e:
                logger.warning("[Planner] Skipping malformed task %s: %s", item, e)

        if not sub_tasks:
            # Fallback: treat the original query as a single task
            sub_tasks = [SubTask(id=1, question=state.original_query, priority=1)]

        logger.info("[Planner] Created %d sub-tasks", len(sub_tasks))
        return {"sub_tasks": sub_tasks}

    except Exception as exc:
        logger.error("[Planner] Failed: %s", exc)
        fallback = [SubTask(id=1, question=state.original_query, priority=1)]
        return {"sub_tasks": fallback, "error": str(exc)}
