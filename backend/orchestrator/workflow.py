"""LangGraph multi-agent orchestration workflow."""
import os
import sys
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
#  Node functions                                                      #
# ------------------------------------------------------------------ #

def _safe_run(agent_key: str, state: ProductStrategyState) -> ProductStrategyState:
    try:
        return _get(agent_key).run(dict(state))
    except Exception as exc:
        state["error"] = f"{agent_key} error: {exc}"
        return state


def customer_feedback_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Customer Feedback Agent..."
    return _safe_run("customer_feedback", state)


def market_research_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Market Research Agent..."
    return _safe_run("market_research", state)


def competitor_analysis_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Competitor Analysis Agent..."
    return _safe_run("competitor_analysis", state)


def swot_analysis_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running SWOT Analysis Agent..."
    return _safe_run("swot_analysis", state)


def feature_prioritization_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Feature Prioritization Agent..."
    return _safe_run("feature_prioritization", state)


def strategy_recommendation_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Strategy Recommendation Agent..."
    return _safe_run("strategy_recommendation", state)


def executive_report_node(state: ProductStrategyState) -> ProductStrategyState:
    state["current_step"] = "Running Executive Report Agent..."
    return _safe_run("executive_report", state)


def chat_node(state: ProductStrategyState) -> ProductStrategyState:
    """RAG-powered conversational node."""
    state["current_step"] = "Generating chat response..."
    try:
        store = ChromaVectorStore()
        query = state.get("query", "")
        history = state.get("chat_history", [])

        # Retrieve relevant document chunks
        doc_ctx = store.get_context(query, k=6)

        # Include summaries from any completed analyses
        analysis_ctx = ""
        for key in [
            "customer_insights", "market_research", "competitor_analysis",
            "swot_analysis", "feature_priorities", "strategy_recommendations",
        ]:
            val = (state.get(key) or "").strip()
            if val:
                analysis_ctx += f"\n\n[{key.replace('_', ' ').title()}]\n{val[:400]}..."

        # Build recent conversation history (last 6 turns)
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

        user_content = f"""Document Context:
{doc_ctx if doc_ctx else "No documents uploaded yet."}

Analysis Results:
{analysis_ctx if analysis_ctx else "No analysis run yet."}

Conversation History:
{history_str if history_str else "No prior messages."}

User Question: {query}"""

        llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, temperature=0.4, http_client=_HTTP_CLIENT)
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
    return "chat" if state.get("mode") == "chat" else "customer_feedback"


# ------------------------------------------------------------------ #
#  Graph construction                                                  #
# ------------------------------------------------------------------ #

def _build_graph():
    g = StateGraph(ProductStrategyState)

    # Register nodes
    g.add_node("customer_feedback", customer_feedback_node)
    g.add_node("market_research", market_research_node)
    g.add_node("competitor_analysis", competitor_analysis_node)
    g.add_node("swot_analysis", swot_analysis_node)
    g.add_node("feature_prioritization", feature_prioritization_node)
    g.add_node("strategy_recommendation", strategy_recommendation_node)
    g.add_node("executive_report", executive_report_node)
    g.add_node("chat", chat_node)

    # Entry-point routing
    g.add_conditional_edges(
        START,
        _router,
        {
            "chat": "chat",
            "customer_feedback": "customer_feedback",
        },
    )

    # Sequential analysis pipeline
    g.add_edge("customer_feedback", "market_research")
    g.add_edge("market_research", "competitor_analysis")
    g.add_edge("competitor_analysis", "swot_analysis")
    g.add_edge("swot_analysis", "feature_prioritization")
    g.add_edge("feature_prioritization", "strategy_recommendation")
    g.add_edge("strategy_recommendation", "executive_report")
    g.add_edge("executive_report", END)

    # Chat exits immediately
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
        current_step="Initialising",
        error=None,
    )


def run_analysis(seed: Optional[dict] = None) -> dict:
    """Run the full 7-agent analysis pipeline and return the final state."""
    state = _blank_state()
    if seed:
        state.update(seed)
    state["mode"] = "analyze"
    return _get_graph().invoke(state)


def run_chat(query: str, chat_history: List[dict], analysis_state: Optional[dict] = None) -> dict:
    """Run the RAG chat node and return the updated state."""
    state = _blank_state()
    if analysis_state:
        # Carry forward completed analysis outputs
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
