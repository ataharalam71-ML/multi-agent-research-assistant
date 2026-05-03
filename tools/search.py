"""
tools/search.py — Web search, arXiv, and Wikipedia tools for the Searcher agent.
Each function is decorated with @tool so LangChain can bind them to an LLM.
"""
from __future__ import annotations
import json
import urllib.parse
import urllib.request
from typing import Any

from langchain_core.tools import tool
from tavily import TavilyClient

from config.settings import get_settings
from models import SearchResult


# ── Tavily web search ──────────────────────────────────────────────────────────

def _tavily_client() -> TavilyClient:
    return TavilyClient(api_key=get_settings().tavily_api_key)


@tool
def web_search(query: str) -> list[dict]:
    """
    Search the web for current information using Tavily.
    Returns a list of {title, url, content, score} dicts.
    Use this for recent news, facts, or general information.
    """
    settings = get_settings()
    client = _tavily_client()
    try:
        response = client.search(
            query=query,
            max_results=settings.max_search_results,
            include_answer=False,
            include_raw_content=False,
        )
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", "")[:1500],  # cap to save tokens
                "score":   round(r.get("score", 0.0), 3),
            }
            for r in response.get("results", [])
        ]
    except Exception as exc:
        return [{"error": str(exc), "title": "", "url": "", "content": "", "score": 0.0}]


# ── arXiv academic search ──────────────────────────────────────────────────────

@tool
def arxiv_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search arXiv for academic papers.
    Use this when the query involves research, ML, science, or technical topics.
    Returns a list of {title, url, summary, authors} dicts.
    """
    encoded = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}&start=0&max_results={max_results}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read().decode()

        # Minimal XML parse (no external deps)
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(raw)
        results = []
        for entry in root.findall("atom:entry", ns):
            title   = (entry.findtext("atom:title",   "", ns) or "").strip()
            summary = (entry.findtext("atom:summary", "", ns) or "").strip()[:800]
            link    = entry.find("atom:id", ns)
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            results.append({
                "title":   title,
                "url":     link.text if link is not None else "",
                "summary": summary,
                "authors": authors[:3],
            })
        return results
    except Exception as exc:
        return [{"error": str(exc)}]


# ── Wikipedia summary ──────────────────────────────────────────────────────────

@tool
def wikipedia_summary(topic: str) -> dict:
    """
    Fetch a short Wikipedia summary for a well-known topic.
    Use this for background context on entities, concepts, or events.
    Returns {title, summary, url}.
    """
    encoded = urllib.parse.quote(topic.replace(" ", "_"))
    api_url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    )
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "MultiAgentResearch/1.0 (educational project)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict[str, Any] = json.loads(resp.read())
        return {
            "title":   data.get("title", ""),
            "summary": data.get("extract", "")[:1000],
            "url":     data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except Exception as exc:
        return {"error": str(exc), "title": "", "summary": "", "url": ""}


# ── Tool registry (imported by agents) ────────────────────────────────────────

ALL_TOOLS = [web_search, arxiv_search, wikipedia_summary]
