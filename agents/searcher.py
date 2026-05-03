"""
agents/searcher.py — Executes web search, arXiv, and Wikipedia calls for
each sub-task produced by the Planner. Stores all results in the graph state.
"""
from __future__ import annotations
import logging
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser

from config.llm import get_llm
from models import ResearchState, SearchResult, RouteDecision
from tools.search import ALL_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research specialist. For each sub-question, choose the best tool(s):
- web_search        → current news, facts, general information
- arxiv_search      → ML papers, science, academic topics
- wikipedia_summary → background context on well-known entities

Call ALL tools that would help. Be thorough — the Writer depends on your output.
"""


def searcher_node(state: ResearchState) -> dict:
    """
    LangGraph node. Reads state.sub_tasks, writes state.search_results.
    """
    logger.info("[Searcher] Running %d sub-tasks (iteration %d)",
                len(state.sub_tasks), state.search_iterations + 1)

    llm = get_llm(temperature=0.0).bind_tools(ALL_TOOLS)
    accumulated_results: list[SearchResult] = list(state.search_results)

    for task in sorted(state.sub_tasks, key=lambda t: t.priority):
        logger.info("[Searcher] → Sub-task %d: %s", task.id, task.question)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Sub-question: {task.question}"),
        ]

        # Tool-use loop (max 3 rounds per sub-task)
        for _ in range(3):
            response = llm.invoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", [])
            if not tool_calls:
                break  # no more tool calls requested

            # Execute each tool call
            for call in tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_fn   = next((t for t in ALL_TOOLS if t.name == tool_name), None)

                if tool_fn is None:
                    logger.warning("[Searcher] Unknown tool: %s", tool_name)
                    continue

                try:
                    raw_results = tool_fn.invoke(tool_args)
                except Exception as exc:
                    logger.error("[Searcher] Tool %s failed: %s", tool_name, exc)
                    raw_results = []

                # Normalise tool output into SearchResult objects
                if isinstance(raw_results, list):
                    for item in raw_results:
                        if isinstance(item, dict) and "error" not in item:
                            accumulated_results.append(SearchResult(
                                query   = task.question,
                                url     = item.get("url", ""),
                                title   = item.get("title", ""),
                                content = item.get("content", item.get("summary", "")),
                                score   = item.get("score", 0.0),
                            ))
                elif isinstance(raw_results, dict) and "error" not in raw_results:
                    accumulated_results.append(SearchResult(
                        query   = task.question,
                        url     = raw_results.get("url", ""),
                        title   = raw_results.get("title", ""),
                        content = raw_results.get("summary", raw_results.get("content", "")),
                        score   = 0.8,
                    ))

                # Feed tool result back to the LLM for the next round
                messages.append(ToolMessage(
                    content=str(raw_results)[:2000],
                    tool_call_id=call["id"],
                ))

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique_results: list[SearchResult] = []
    for r in accumulated_results:
        if r.url and r.url not in seen_urls:
            seen_urls.add(r.url)
            unique_results.append(r)

    logger.info("[Searcher] Collected %d unique results", len(unique_results))
    return {
        "search_results":     unique_results,
        "search_iterations":  state.search_iterations + 1,
    }
