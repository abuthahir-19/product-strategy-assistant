import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

_SYSTEM = """You are a Chief of Staff and Executive Communications expert.
Distil complex analysis into a crisp, decisive executive summary written for C-suite audiences.

Executive summaries must be:
• Concise (500–700 words)
• Action-oriented — every section drives a decision
• Free of jargon
• Clear on business impact and urgency
• Structured for fast scanning"""


class ExecutiveReportAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Executive Report Agent",
            description="Produces the executive summary and consolidates the final report package",
        )

    def run(self, state: dict) -> dict:
        snippets = []
        labels = [
            ("Customer Insights", "customer_insights"),
            ("Market Research", "market_research"),
            ("Competitor Analysis", "competitor_analysis"),
            ("SWOT Analysis", "swot_analysis"),
            ("Feature Priorities", "feature_priorities"),
            ("Strategic Recommendations", "strategy_recommendations"),
        ]
        for label, key in labels:
            val = (state.get(key) or "").strip()
            if val:
                snippets.append(f"### {label}\n{val[:450]}...")

        combined = "\n\n".join(snippets) if snippets else "No upstream analysis available."

        prompt = f"""Based on the analysis summaries below, write a compelling **Executive Summary**.

ANALYSIS SUMMARIES:
{combined}

Structure the Executive Summary exactly as follows:

## Executive Summary

### Situation
(2–3 sentences: current state of the business / product)

### Key Findings
(5 bullet points — most important insights across all analyses)

### Critical Opportunities
(Top 3 opportunities with estimated business impact)

### Key Risks
(Top 3 risks requiring immediate attention)

### Strategic Priorities
(The 3 most important moves leadership must make)

### Recommended Actions
- **Immediate (Next 30 days):** [3 specific actions]
- **Short-term (30–90 days):** [3 specific actions]
- **Medium-term (3–6 months):** [3 specific actions]

### Expected Business Impact
(Projected outcomes: revenue, NPS, retention, market share)

Write at a C-suite level. Be direct, decisive, and impact-focused.
Limit to 600 words maximum."""

        result = self.llm_call(_SYSTEM, prompt)
        state["executive_summary"] = result
        state["current_step"] = "Executive Report complete"
        return state
