import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

_SYSTEM = """You are a Principal Product Manager expert in feature prioritisation frameworks.
Apply rigorous, data-driven methods to rank features by customer value and business impact.

Frameworks to apply:
• RICE scoring (Reach × Impact × Confidence ÷ Effort)
• MoSCoW classification (Must / Should / Could / Won't)
• Value vs. Effort matrix for quick wins
• Customer-demand weighting from feedback data"""


class FeaturePrioritizationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Feature Prioritization Agent",
            description="Prioritises features using RICE/MoSCoW frameworks from customer & market data",
        )

    def run(self, state: dict) -> dict:
        ctx = self.retrieve(
            "feature requests improvements enhancements product backlog priority customer asks",
            k=8,
        )

        customer = (state.get("customer_insights") or "")[:700]
        market = (state.get("market_research") or "")[:500]
        swot = (state.get("swot_analysis") or "")[:500]

        prompt = f"""Based on the data below, produce a **Feature Prioritization Recommendations** report.

CUSTOMER INSIGHTS:
{customer or "Not available."}

MARKET RESEARCH:
{market or "Not available."}

SWOT ANALYSIS:
{swot or "Not available."}

FEATURE REQUEST DATA FROM DOCUMENTS:
{ctx or "Not available."}

Structure your report exactly as follows:

## Priority Tier 1 — Must Have (Q1–Q2)
For each feature:
- **Feature name**: [name]
  - Customer need addressed:
  - Business impact:
  - Estimated effort: Low / Medium / High
  - RICE rationale:
(List 4–6 features)

## Priority Tier 2 — Should Have (Q2–Q3)
(3–5 features, same sub-structure)

## Priority Tier 3 — Could Have (Q3–Q4)
(3–4 features, same sub-structure)

## Priority Tier 4 — Future Consideration / Won't Have Now
(2–3 features with one-line rationale)

## Quick Wins (High Value, Low Effort)
(2–3 features that can ship fast for outsized impact)

## Prioritisation Rationale
(Brief paragraph explaining the scoring logic used)

Map each feature back to a specific customer complaint, request, or market signal wherever possible."""

        result = self.llm_call(_SYSTEM, prompt)
        state["feature_priorities"] = result
        state["current_step"] = "Feature Prioritization complete"
        return state
