import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

_SYSTEM = """You are a Strategic Business Analyst specialising in SWOT frameworks and strategic synthesis.
Synthesise insights from customer feedback, market data, and competitive intelligence into a rigorous SWOT analysis.

A great SWOT:
• Uses specific, data-backed points — no generic filler
• Prioritises the most impactful factors (5–7 per quadrant)
• Connects internal capabilities to external dynamics
• Concludes with clear strategic implications (SO, ST, WO, WT strategies)"""


class SWOTAnalysisAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="SWOT Analysis Agent",
            description="Creates a comprehensive SWOT analysis by synthesising all upstream agent outputs",
        )

    def run(self, state: dict) -> dict:
        customer = (state.get("customer_insights") or "")[:900]
        market = (state.get("market_research") or "")[:900]
        competitor = (state.get("competitor_analysis") or "")[:900]

        extra_ctx = self.retrieve("strengths weaknesses opportunities threats strategy", k=4)

        prompt = f"""Synthesise the following analyses into a comprehensive **SWOT Analysis**.

CUSTOMER INSIGHTS:
{customer or "Not available."}

MARKET RESEARCH:
{market or "Not available."}

COMPETITOR ANALYSIS:
{competitor or "Not available."}

ADDITIONAL CONTEXT:
{extra_ctx or "None."}

Produce the SWOT with the exact structure below:

## STRENGTHS (Internal Positives)
(5–7 specific, data-backed strengths)
- Strength 1: ...
- Strength 2: ...

## WEAKNESSES (Internal Negatives)
(5–7 specific weaknesses or gaps to address)
- Weakness 1: ...

## OPPORTUNITIES (External Positives)
(5–7 market and product opportunities)
- Opportunity 1: ...

## THREATS (External Negatives)
(5–7 external risks)
- Threat 1: ...

## STRATEGIC IMPLICATIONS
### SO Strategies — Leverage strengths to capture opportunities
- ...
### ST Strategies — Use strengths to mitigate threats
- ...
### WO Strategies — Fix weaknesses to seize opportunities
- ...
### WT Strategies — Minimise weaknesses and avoid threats
- ...

Every bullet must be specific and traceable to the data provided."""

        result = self.llm_call(_SYSTEM, prompt)
        state["swot_analysis"] = result
        state["current_step"] = "SWOT Analysis complete"
        return state
