"""
evals/eval_suite.py — Automated evaluation of the research pipeline.

Measures:
  - Answer quality (LLM-as-judge, 0-10)
  - Citation presence (does the answer cite sources?)
  - Latency (seconds per query)
  - Token usage (approximate)

Run with:  python -m evals.eval_suite
"""
from __future__ import annotations
import json
import logging
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, asdict
from langchain_core.messages import SystemMessage, HumanMessage

from config.llm import get_llm
from graph import get_graph
from models import ResearchState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)  # quiet during evals

# ── Test questions (20 total, 3 difficulty levels) ────────────────────────────

TEST_QUESTIONS = [
    # Easy (5)
    {"id": 1, "query": "What is the capital of Australia?",                    "difficulty": "easy"},
    {"id": 2, "query": "Who invented the telephone?",                           "difficulty": "easy"},
    {"id": 3, "query": "What is photosynthesis?",                               "difficulty": "easy"},
    {"id": 4, "query": "What does GPU stand for?",                              "difficulty": "easy"},
    {"id": 5, "query": "What year did World War II end?",                       "difficulty": "easy"},
    # Medium (10)
    {"id": 6,  "query": "How does transformer architecture work in LLMs?",      "difficulty": "medium"},
    {"id": 7,  "query": "What are the main causes of inflation in 2024?",       "difficulty": "medium"},
    {"id": 8,  "query": "Explain the difference between RAG and fine-tuning",   "difficulty": "medium"},
    {"id": 9,  "query": "What is CRISPR and how is it used in medicine?",       "difficulty": "medium"},
    {"id": 10, "query": "How do vector databases work?",                         "difficulty": "medium"},
    {"id": 11, "query": "What is the current state of quantum computing?",      "difficulty": "medium"},
    {"id": 12, "query": "Explain how RLHF is used to train language models",    "difficulty": "medium"},
    {"id": 13, "query": "What are the environmental impacts of data centers?",  "difficulty": "medium"},
    {"id": 14, "query": "How does the TCP/IP protocol work?",                   "difficulty": "medium"},
    {"id": 15, "query": "What are the pros and cons of microservices?",         "difficulty": "medium"},
    # Hard (5)
    {"id": 16, "query": "Compare the long-term risks of AGI vs narrow AI",     "difficulty": "hard"},
    {"id": 17, "query": "What are the economic implications of automation on employment in India?", "difficulty": "hard"},
    {"id": 18, "query": "How should AI regulation differ between the US and EU, and why?", "difficulty": "hard"},
    {"id": 19, "query": "Analyse the trade-offs in distributed systems: CAP theorem", "difficulty": "hard"},
    {"id": 20, "query": "What is the current state of nuclear fusion research and its timeline?", "difficulty": "hard"},
]


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    id: int
    query: str
    difficulty: str
    answer: str
    quality_score: int      # LLM-as-judge (0-10)
    has_citations: bool
    confidence_score: int   # critic's score
    latency_seconds: float
    search_iterations: int
    critic_retries: int
    error: str = ""


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an objective evaluator. Rate the quality of the research answer below.

Query: {query}
Answer: {answer}

Score from 0-10:
- 9-10: Comprehensive, accurate, well-cited, clear
- 7-8:  Good coverage, mostly accurate, readable
- 5-6:  Partial coverage, some inaccuracies or vagueness
- 3-4:  Missing key information or contains errors
- 0-2:  Wrong, irrelevant, or refused to answer

Return ONLY a JSON object: {{"score": <0-10>, "reason": "<one sentence>"}}
"""


def judge_answer(query: str, answer: str) -> tuple[int, str]:
    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content="You are a strict quality evaluator."),
        HumanMessage(content=JUDGE_PROMPT.format(query=query, answer=answer[:2000])),
    ]
    try:
        response = llm.invoke(messages)

        # Handle Gemini returning list of content blocks
        raw = response.content
        if isinstance(raw, list):
            raw = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            )

        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        return int(data["score"]), data.get("reason", "")
    except Exception as exc:
        logger.warning("Judge failed: %s", exc)
        return 0, str(exc)


# ── Single eval run ────────────────────────────────────────────────────────────

def run_single(question: dict) -> EvalResult:
    graph = get_graph()
    start = time.time()

    try:
        initial = ResearchState(original_query=question["query"])
        result  = graph.invoke(initial)
        state   = ResearchState(**result)
        answer  = state.final_answer or state.draft
        error   = state.error or ""
    except Exception as exc:
        elapsed = time.time() - start
        return EvalResult(
            id=question["id"], query=question["query"],
            difficulty=question["difficulty"], answer="", quality_score=0,
            has_citations=False, confidence_score=0, latency_seconds=elapsed,
            search_iterations=0, critic_retries=0, error=str(exc),
        )

    elapsed = time.time() - start
    quality, _ = judge_answer(question["query"], answer)

    return EvalResult(
        id                 = question["id"],
        query              = question["query"],
        difficulty         = question["difficulty"],
        answer             = answer[:300],  # truncate for storage
        quality_score      = quality,
        has_citations      = "[1]" in answer or "http" in answer,
        confidence_score   = state.confidence_score,
        latency_seconds    = round(elapsed, 2),
        search_iterations  = state.search_iterations,
        critic_retries     = state.critic_retries,
        error              = error,
    )


# ── Full suite ─────────────────────────────────────────────────────────────────

def run_suite(subset: list[int] | None = None) -> list[EvalResult]:
    questions = TEST_QUESTIONS
    if subset:
        questions = [q for q in questions if q["id"] in subset]

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['difficulty'].upper()} — {q['query'][:60]}...")
        r = run_single(q)
        results.append(r)
        status = "✅" if r.quality_score >= 7 else "⚠️" if r.quality_score >= 5 else "❌"
        print(f"         {status} quality={r.quality_score}/10  confidence={r.confidence_score}/10  latency={r.latency_seconds}s")

    return results


def print_summary(results: list[EvalResult]) -> None:
    total = len(results)
    avg_quality     = sum(r.quality_score for r in results) / total
    avg_confidence  = sum(r.confidence_score for r in results) / total
    avg_latency     = sum(r.latency_seconds for r in results) / total
    citation_rate   = sum(1 for r in results if r.has_citations) / total * 100
    error_rate      = sum(1 for r in results if r.error) / total * 100

    print("\n" + "="*60)
    print("EVAL SUMMARY")
    print("="*60)
    print(f"  Questions evaluated : {total}")
    print(f"  Avg quality (judge) : {avg_quality:.1f}/10")
    print(f"  Avg confidence      : {avg_confidence:.1f}/10")
    print(f"  Avg latency         : {avg_latency:.1f}s")
    print(f"  Citation rate       : {citation_rate:.0f}%")
    print(f"  Error rate          : {error_rate:.0f}%")

    by_difficulty: dict[str, list] = {}
    for r in results:
        by_difficulty.setdefault(r.difficulty, []).append(r.quality_score)

    print("\n  By difficulty:")
    for diff, scores in by_difficulty.items():
        print(f"    {diff:6s}: avg {sum(scores)/len(scores):.1f}/10  ({len(scores)} questions)")
    print("="*60)

    # Save to JSON
    output_path = "evals/results.json"
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    # Run a quick subset (5 questions) by default; pass --all for the full suite
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all 20 questions")
    parser.add_argument("--ids", nargs="+", type=int, help="Run specific question IDs")
    args = parser.parse_args()

    if args.all:
        results = run_suite()
    elif args.ids:
        results = run_suite(subset=args.ids)
    else:
        # Quick 5-question demo
        results = run_suite(subset=[1, 6, 10, 16, 20])

    print_summary(results)
