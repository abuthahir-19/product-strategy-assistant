import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

_SYSTEM = """You are an expert Competitive Intelligence Analyst.
Your job is to map the competitive landscape, identify gaps, and surface strategic positioning insights.

Focus on:
• Direct and indirect competitors
• Competitor strengths and weaknesses
• Feature and pricing gaps
• Potential threats and market disruptions
• White-space opportunities our product can exploit

If dedicated competitor documents are unavailable, derive competitive insights from
customer feedback (what customers compare us to), market data, and product positioning signals."""


class CompetitorAnalysisAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Competitor Analysis Agent",
            description="Analyses competitive landscape and market positioning",
        )

    def run(self, state: dict) -> dict:
        ctx = self.retrieve(
            "competitor analysis competitive landscape market positioning product comparison pricing",
            k=8,
        )
        if not ctx:
            ctx = "No dedicated competitor documents uploaded. Using cross-agent signals."

        customer_snip = (state.get("customer_insights") or "")[:600]
        market_snip = (state.get("market_research") or "")[:600]

        prompt = f"""Based on all available information, produce a **Competitor Analysis Report**.

COMPETITOR / MARKET DATA:
{ctx}

CUSTOMER INSIGHTS SUMMARY:
{customer_snip or "Not yet available."}

MARKET RESEARCH SUMMARY:
{market_snip or "Not yet available."}

Structure your report exactly as follows:

## 1. Competitive Landscape Overview
(Market structure: fragmented / consolidated, key players)

## 2. Key Competitors
(For each: name, positioning, estimated strength — 3–5 competitors)

## 3. Competitor Strengths
(Where rivals excel — features, pricing, UX, distribution)

## 4. Competitor Weaknesses & Gaps
(Unmet needs, poor reviews, missing features)

## 5. Our Competitive Advantages
(Derived from customer ratings, product performance, unique attributes)

## 6. Competitive Threats
(Risks from rivals or new entrants)

## 7. Exploitable Opportunities
(White-space, underserved segments, positioning gaps)

## 8. Differentiation Recommendations
(3–5 concrete ways to strengthen competitive position)

If dedicated competitor data is scarce, draw on customer sentiment and market data to infer positioning."""

        result = self.llm_call(_SYSTEM, prompt)
        state["competitor_analysis"] = result
        state["current_step"] = "Competitor Analysis complete"
        return state
