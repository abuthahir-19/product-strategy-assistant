"""LangGraph multi-agent orchestration workflow."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from vector_store.chroma_store import ChromaVectorStore

# Shared httpx client — skips SSL cert verification for the course gateway
_HTTP_CLIENT = httpx.Client(verify=False)

# ------------------------------------------------------------------ #
#  Shared state schema                                                 #
# ------------------------------------------------------------------ #

class ProductStrategyState(TypedDict):
    # Routing
    mode: str  # "analyze" | "chat"

    # Chat
    query: Optional[str]
    chat_history: List[dict]
    chat_response: Optional[str]

    # Agent outputs
    customer_insights: Optional[str]
    market_research: Optional[str]
    competitor_analysis: Optional[str]
    swot_analysis: Optional[str]
    feature_priorities: Optional[str]
    strategy_recommendations: Optional[str]
    executive_summary: Optional[str]

    # Metadata
    current_step: str
    error: Optional[str]


# ------------------------------------------------------------------ #
#  Lazy agent registry                                                 #
# ------------------------------------------------------------------ #
_agents: dict = {}


def _get(name: str):
    if name not in _agents:
        if name == "customer_feedback":
            from agents.customer_feedback_agent import CustomerFeedbackAgent
            _agents[name] = CustomerFeedbackAgent()
        elif name == "market_research":
            from agents.market_research_agent import MarketResearchAgent
            _agents[name] = MarketResearchAgent()
        elif name == "competitor_analysis":
            from agents.competitor_analysis_agent import CompetitorAnalysisAgent
            _agents[name] = CompetitorAnalysisAgent()
        elif name == "swot_analysis":
            from agents.swot_analysis_agent import SWOTAnalysisAgent
            _agents[name] = SWOTAnalysisAgent()
        elif name == "feature_prioritization":
            from agents.feature_prioritization_agent import FeaturePrioritizationAgent
            _agents[name] = FeaturePrioritizationAgent()
        elif name == "strategy_recommendation":
            from agents.strategy_recommendation_agent import StrategyRecommendationAgent
            _agents[name] = StrategyRecommendationAgent()
        elif name == "executive_report":
            from agents.executive_report_agent import ExecutiveReportAgent
            _agents[name] = ExecutiveReportAgent()
    return _agents[name]


# ------------------------------------------------------------------ #
#  Safe runner                                                         #
# ------------------------------------------------------------------ #

def _safe_run(agent_key: str, state: dict) -> dict:
    try:
        return _get(agent_key).run(dict(state))
    except Exception as exc:
        state = dict(state)
        state["error"] = f"{agent_key} error: {exc}"
        return state


# ------------------------------------------------------------------ #
#  Node 1 — Parallel: Customer Feedback + Market Research +           #
#            Competitor Analysis (independent, no shared inputs)      #
# ------------------------------------------------------------------ #

# Maps agent key → output field written by that agent
_PARALLEL_AGENTS = {
    "customer_feedback":  "customer_insights",
    "market_research":    "market_research",
    "competitor_analysis":"competitor_analysis",
}


def parallel_analysis_node(state: ProductStrategyState) -> ProductStrategyState:
    """
    Run the first three agents concurrently using a thread pool.
    They share nothing from each other, so parallelism is safe.
    Cuts this phase from ~3 sequential calls to 1 concurrent batch.
    """
    state["current_step"] = (
        "Running Customer Feedback, Market Research & "
        "Competitor Analysis in parallel…"
    )

    def _run(agent_key: str) -> dict:
        return _safe_run(agent_key, state)

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_map = {pool.submit(_run, key): key for key in _PARALLEL_AGENTS}
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = {"error": str(exc)}

    # Merge each agent's output field back into the shared state
    for agent_key, output_field in _PARALLEL_AGENTS.items():
        value = results.get(agent_key, {}).get(output_field)
        if value:
            state[output_field] = value

    # Collect any errors
    errors = [
        f"{k}: {r['error']}"
        for k, r in results.items()
        if r.get("error")
    ]
    if errors:
        state["error"] = " | ".join(errors)

    state["current_step"] = "Phase 1 complete — running SWOT Analysis…"
    return state


# ------------------------------------------------------------------ #
#  Nodes 2-5 — Sequential (each depends on upstream outputs)         #
# ------------------------------------------------------------------ #

def swot_analysis_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running SWOT Analysis Agent…"
    return _safe_run("swot_analysis", state)


def feature_prioritization_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Feature Prioritization Agent…"
    return _safe_run("feature_prioritization", state)


def strategy_recommendation_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Strategy Recommendation Agent…"
    return _safe_run("strategy_recommendation", state)


def executive_report_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Executive Report Agent…"
    return _safe_run("executive_report", state)


# ------------------------------------------------------------------ #
#  Chat node                                                           #
# ------------------------------------------------------------------ #

def chat_node(state: ProductStrategyState) -> ProductStrategyState:
    """RAG-powered conversational node."""
    state["current_step"] = "Generating chat response…"
    try:
        store = ChromaVectorStore()
        query = state.get("query", "")
        history = state.get("chat_history", [])

        doc_ctx = store.get_context(query, k=4)

        analysis_ctx = ""
        for key in [
            "customer_insights", "market_research", "competitor_analysis",
            "swot_analysis", "feature_priorities", "strategy_recommendations",
        ]:
            val = (state.get(key) or "").strip()
            if val:
                analysis_ctx += f"\n\n[{key.replace('_', ' ').title()}]\n{val[:350]}…"

        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in history[-6:]
        )

        system_prompt = (
            "You are an AI Product Strategy Assistant. "
            "Answer questions based strictly on the provided document context and analysis results. "
            "Be specific, reference data when possible, and admit when information is unavailable. "
            "Keep answers concise but complete."
        )

        user_content = (
            f"Document Context:\n{doc_ctx or 'No documents uploaded yet.'}\n\n"
            f"Analysis Results:\n{analysis_ctx or 'No analysis run yet.'}\n\n"
            f"Conversation History:\n{history_str or 'No prior messages.'}\n\n"
            f"User Question: {query}"
        )

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.4,
            http_client=_HTTP_CLIENT,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        state["chat_response"] = response.content
    except Exception as exc:
        state["chat_response"] = f"Error generating response: {exc}"
        state["error"] = str(exc)
    return state


# ------------------------------------------------------------------ #
#  Router                                                              #
# ------------------------------------------------------------------ #

def _router(state: ProductStrategyState) -> str:
    return "chat" if state.get("mode") == "chat" else "parallel_analysis"


# ------------------------------------------------------------------ #
#  Graph construction                                                  #
# ------------------------------------------------------------------ #

def _build_graph():
    g = StateGraph(ProductStrategyState)

    g.add_node("parallel_analysis",      parallel_analysis_node)
    g.add_node("swot_analysis",          swot_analysis_node)
    g.add_node("feature_prioritization", feature_prioritization_node)
    g.add_node("strategy_recommendation",strategy_recommendation_node)
    g.add_node("executive_report",       executive_report_node)
    g.add_node("chat",                   chat_node)

    g.add_conditional_edges(
        START,
        _router,
        {
            "chat":              "chat",
            "parallel_analysis": "parallel_analysis",
        },
    )

    # Analysis pipeline — parallel phase then sequential synthesis
    g.add_edge("parallel_analysis",       "swot_analysis")
    g.add_edge("swot_analysis",           "feature_prioritization")
    g.add_edge("feature_prioritization",  "strategy_recommendation")
    g.add_edge("strategy_recommendation", "executive_report")
    g.add_edge("executive_report",        END)

    g.add_edge("chat", END)

    return g.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def _blank_state() -> ProductStrategyState:
    return ProductStrategyState(
        mode="analyze",
        query=None,
        chat_history=[],
        chat_response=None,
        customer_insights=None,
        market_research=None,
        competitor_analysis=None,
        swot_analysis=None,
        feature_priorities=None,
        strategy_recommendations=None,
        executive_summary=None,
        current_step="Initialising…",
        error=None,
    )


def run_analysis(seed: Optional[dict] = None) -> dict:
    """Run the full pipeline and return the final state."""
    state = _blank_state()
    if seed:
        state.update(seed)
    state["mode"] = "analyze"
    return _get_graph().invoke(state)


def run_chat(query: str, chat_history: List[dict], analysis_state: Optional[dict] = None) -> dict:
    """Run the RAG chat node and return the updated state."""
    state = _blank_state()
    if analysis_state:
        for key in [
            "customer_insights", "market_research", "competitor_analysis",
            "swot_analysis", "feature_priorities", "strategy_recommendations",
            "executive_summary",
        ]:
            if analysis_state.get(key):
                state[key] = analysis_state[key]
    state["mode"] = "chat"
    state["query"] = query
    state["chat_history"] = chat_history
    return _get_graph().invoke(state)
