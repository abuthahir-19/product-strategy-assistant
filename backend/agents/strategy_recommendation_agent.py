import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

_SYSTEM = """You are a Chief Product Officer and Strategic Advisor.
Synthesise all analytical outputs into a clear, time-bound, executable Strategic Action Plan.

Great strategy plans are:
• Specific and time-bound (30/60/90 day cadence)
• Ranked by business impact
• Realistic and resource-aware
• Measurable (KPIs for each initiative)
• Narratively coherent — each section builds on the last"""


class StrategyRecommendationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Strategy Recommendation Agent",
            description="Synthesises all analyses into a strategic action plan and product roadmap",
        )

    def run(self, state: dict) -> dict:
        customer = (state.get("customer_insights") or "")[:600]
        market = (state.get("market_research") or "")[:600]
        competitor = (state.get("competitor_analysis") or "")[:500]
        swot = (state.get("swot_analysis") or "")[:500]
        features = (state.get("feature_priorities") or "")[:500]

        extra_ctx = self.retrieve("strategy roadmap planning business objectives OKR", k=4)

        prompt = f"""Synthesise the analyses below into a **Strategic Recommendations & Action Plan**.

CUSTOMER INSIGHTS:
{customer or "Not available."}

MARKET RESEARCH:
{market or "Not available."}

COMPETITOR ANALYSIS:
{competitor or "Not available."}

SWOT ANALYSIS:
{swot or "Not available."}

FEATURE PRIORITIES:
{features or "Not available."}

ADDITIONAL CONTEXT:
{extra_ctx or "None."}

Structure your plan exactly as follows:

## Strategic Vision (12-Month Outlook)
(One clear strategic direction statement)

## Strategic Priority 1: [Title]
- Objective:
- Key Actions (3–4 bullets):
- Owner / Function:
- Success Metric:
- Timeline:

## Strategic Priority 2: [Title]
(same sub-structure)

## Strategic Priority 3: [Title]
(same sub-structure)

## Product Roadmap Suggestions
### Q1 — Foundation
- Key deliverables:
### Q2 — Growth
- Key deliverables:
### Q3 — Scale
- Key deliverables:
### Q4 — Optimise
- Key deliverables:

## Resource & Investment Focus
(Where to concentrate budget and headcount for maximum ROI)

## Risk Mitigation Plan
(Top 3 risks with mitigation approach)

## KPIs & Success Metrics
(5–7 measurable KPIs to track progress)

## 30 / 60 / 90 Day Action Plan
### First 30 Days (Immediate actions)
- ...
### Days 31–60 (Short-term milestones)
- ...
### Days 61–90 (Quarterly goals)
- ...

Be specific, bold, and data-driven throughout."""

        result = self.llm_call(_SYSTEM, prompt)
        state["strategy_recommendations"] = result
        state["current_step"] = "Strategy Recommendations complete"
        return state
