"""
evals/test_agents.py — Unit tests for each agent node using mocked LLMs.
Run with:  pytest evals/test_agents.py -v
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock, patch
from models import ResearchState, SubTask, SearchResult, CritiqueResult, RouteDecision


# ── Planner tests ──────────────────────────────────────────────────────────────

class TestPlannerNode:

    @patch("agents.planner.get_llm")
    def test_creates_sub_tasks(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.return_value.invoke.return_value = MagicMock(
            content='[{"id":1,"question":"What is X?","priority":1},{"id":2,"question":"Why X?","priority":2}]'
        )
        mock_get_llm.return_value = mock_llm.return_value

        from agents.planner import planner_node
        state = ResearchState(original_query="Explain X")
        result = planner_node(state)

        assert "sub_tasks" in result
        assert len(result["sub_tasks"]) == 2
        assert result["sub_tasks"][0].question == "What is X?"

    @patch("agents.planner.get_llm")
    def test_fallback_on_invalid_json(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.return_value.invoke.return_value = MagicMock(content="not json at all")
        mock_get_llm.return_value = mock_llm.return_value

        from agents.planner import planner_node
        state = ResearchState(original_query="Fallback query")
        result = planner_node(state)

        # Should fall back to single task with original query
        assert len(result["sub_tasks"]) == 1
        assert result["sub_tasks"][0].question == "Fallback query"


# ── Writer tests ───────────────────────────────────────────────────────────────

class TestWriterNode:

    @patch("agents.writer.get_llm")
    def test_produces_draft(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="## Answer\nThis is the answer [1].\n## Sources\n[1] Example — http://example.com"
        )
        mock_get_llm.return_value = mock_llm.return_value

        from agents.writer import writer_node
        state = ResearchState(
            original_query="What is X?",
            search_results=[
                SearchResult(query="What is X?", url="http://example.com",
                             title="Example", content="X is a thing", score=0.9)
            ]
        )
        result = writer_node(state)

        assert "draft" in result
        assert len(result["draft"]) > 0
        assert "sources" in result

    @patch("agents.writer.get_llm")
    def test_incorporates_critique(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.return_value.invoke.return_value = MagicMock(content="## Improved\nFixed answer.")
        mock_get_llm.return_value = mock_llm.return_value

        from agents.writer import writer_node
        critique = CritiqueResult(score=4, approved=False,
                                   feedback="Add more detail.", missing_aspects=["context"])
        state = ResearchState(
            original_query="What is X?",
            search_results=[SearchResult(query="X", url="http://a.com", title="A", content="A is A", score=0.8)],
            critique=critique,
        )
        result = writer_node(state)
        assert "draft" in result


# ── Critic tests ───────────────────────────────────────────────────────────────

class TestCriticNode:

    @patch("agents.critic.get_llm")
    def test_approves_good_draft(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.return_value.invoke.return_value = MagicMock(
            content='{"score":9,"approved":true,"hallucination_flags":[],"missing_aspects":[],"feedback":"Excellent answer."}'
        )
        mock_get_llm.return_value = mock_llm.return_value

        from agents.critic import critic_node
        state = ResearchState(
            original_query="What is X?",
            draft="## X\nX is Y [1].\n## Sources\n[1] URL",
            search_results=[SearchResult(query="X", url="http://a.com", title="A", content="A", score=0.9)],
        )
        result = critic_node(state)

        assert result["route"] == RouteDecision.APPROVE
        assert result["final_answer"] != ""

    @patch("agents.critic.get_llm")
    def test_rejects_bad_draft(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.return_value.invoke.return_value = MagicMock(
            content='{"score":3,"approved":false,"hallucination_flags":["claim X"],"missing_aspects":["context"],"feedback":"Rewrite completely."}'
        )
        mock_get_llm.return_value = mock_llm.return_value

        from agents.critic import critic_node
        state = ResearchState(
            original_query="What is X?",
            draft="X is totally Z",
            search_results=[],
        )
        result = critic_node(state)

        assert result["route"] == RouteDecision.WRITE
        assert result["final_answer"] == ""

    @patch("agents.critic.get_llm")
    def test_max_retries_forces_approval(self, mock_get_llm):
        from config.settings import get_settings
        from agents.critic import critic_node
        state = ResearchState(
            original_query="Q",
            draft="D",
            search_results=[],
            critic_retries=get_settings().max_critic_retries,
        )
        result = critic_node(state)
        assert result["route"] == RouteDecision.APPROVE


# ── Memory tests ───────────────────────────────────────────────────────────────

class TestMemoryStore:

    def test_set_and_get(self):
        from memory.store import MemoryStore
        store = MemoryStore()  # no Redis URL → in-memory
        store.set("key1", {"value": 42})
        assert store.get("key1") == {"value": 42}

    def test_get_missing_returns_none(self):
        from memory.store import MemoryStore
        store = MemoryStore()
        assert store.get("nonexistent") is None

    def test_append_to_list(self):
        from memory.store import MemoryStore
        store = MemoryStore()
        store.append_to_list("mylist", "a")
        store.append_to_list("mylist", "b")
        assert store.get("mylist") == ["a", "b"]

    def test_clear(self):
        from memory.store import MemoryStore
        store = MemoryStore()
        store.set("k1", "v1")
        store.set("k2", "v2")
        store.clear()
        assert store.get("k1") is None
