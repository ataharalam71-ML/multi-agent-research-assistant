"""
ui/app.py — Streamlit frontend for the multi-agent research assistant.
Run with:  streamlit run ui/app.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uuid
import time
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from graph import get_graph
from models import ResearchState, RouteDecision
from memory.store import VectorMemory
from config.settings import get_settings

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.agent-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.badge-planner  { background: #EEEDFE; color: #3C3489; }
.badge-searcher { background: #E1F5EE; color: #085041; }
.badge-writer   { background: #FAEEDA; color: #633806; }
.badge-critic   { background: #FCEBEB; color: #791F1F; }
.confidence-bar { height: 8px; border-radius: 4px; background: #e0e0e0; }
.confidence-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #ef5350, #ffa726, #66bb6a); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    show_trace = st.toggle("Show agent trace", value=True)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
1. **Planner** — breaks your query into sub-questions  
2. **Searcher** — calls web/academic APIs  
3. **Writer** — synthesises a cited answer  
4. **Critic** — fact-checks and scores  
""")
    st.divider()
    if st.button("🗑 Clear history"):
        st.session_state.history = []

# ── Session state ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🔬 Multi-Agent Research Assistant")
st.caption("Powered by LangGraph · Planner · Searcher · Writer · Critic")

# Render past conversations
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["query"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{item['score']}/10")
        c2.metric("Sources",    item["sources"])
        c3.metric("Retries",    item["retries"])

# Input
query = st.chat_input("Ask a research question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        trace_container    = st.container() if show_trace else None
        metrics_container  = st.container()

        # ── Stream the graph ───────────────────────────────────────────────────
        graph = get_graph()
        initial_state = ResearchState(original_query=query)

        agent_logs: list[str] = []
        final_state_dict = {}

        # Use LangGraph streaming to get node-by-node updates
        stream_iter = graph.stream(
            initial_state,
            stream_mode="updates",   # yields {node_name: state_delta} per node
        )

        for chunk in stream_iter:
            for node_name, delta in chunk.items():
                # Update trace
                if show_trace and trace_container:
                    badge_class = f"badge-{node_name}"
                    if node_name == "planner" and delta.get("sub_tasks"):
                        tasks = delta["sub_tasks"]
                        log = f'<span class="agent-badge {badge_class}">Planner</span>Created {len(tasks)} sub-tasks'
                        agent_logs.append(log)
                    elif node_name == "searcher" and delta.get("search_results"):
                        count = len(delta["search_results"])
                        log = f'<span class="agent-badge {badge_class}">Searcher</span>Found {count} unique results'
                        agent_logs.append(log)
                    elif node_name == "writer" and delta.get("draft"):
                        chars = len(delta["draft"])
                        log = f'<span class="agent-badge {badge_class}">Writer</span>Draft: {chars} chars'
                        agent_logs.append(log)
                        # Show partial draft immediately
                        answer_placeholder.markdown(delta["draft"][:500] + " ...")
                    elif node_name == "critic" and delta.get("critique"):
                        cr = delta["critique"]
                        score = cr.score if hasattr(cr, "score") else cr.get("score", "?")
                        approved = cr.approved if hasattr(cr, "approved") else cr.get("approved")
                        status = "✅ Approved" if approved else "🔄 Needs revision"
                        log = f'<span class="agent-badge {badge_class}">Critic</span>{status} (score {score}/10)'
                        agent_logs.append(log)

                    with trace_container:
                        st.markdown(
                            "<br>".join(agent_logs),
                            unsafe_allow_html=True,
                        )

                final_state_dict.update(delta)

        # Build final state
        final = ResearchState(**{**initial_state.model_dump(), **final_state_dict})
        answer = final.final_answer or final.draft

        # Show final answer
        answer_placeholder.markdown(answer)

        # Metrics
        with metrics_container:
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence",   f"{final.confidence_score}/10")
            c2.metric("Sources",      len(final.sources))
            c3.metric("Critic retries", final.critic_retries)

            if final.sources:
                with st.expander("📚 Sources"):
                    for i, url in enumerate(final.sources, 1):
                        st.markdown(f"[{i}] {url}")

            if final.critique and show_trace:
                with st.expander("🔍 Critic's evaluation"):
                    cr = final.critique
                    st.write(f"**Score:** {cr.score}/10")
                    st.write(f"**Feedback:** {cr.feedback}")
                    if cr.hallucination_flags:
                        st.warning("Hallucination flags: " + ", ".join(cr.hallucination_flags))
                    if cr.missing_aspects:
                        st.info("Missing aspects: " + ", ".join(cr.missing_aspects))

        # Save to history
        st.session_state.history.append({
            "query":   query,
            "answer":  answer,
            "score":   final.confidence_score,
            "sources": len(final.sources),
            "retries": final.critic_retries,
        })

        # Persist to vector memory
        try:
            vm = VectorMemory(persist_dir=get_settings().chroma_persist_dir)
            vm.save_session(query=query, answer=answer, session_id=str(uuid.uuid4()))
        except Exception:
            pass
