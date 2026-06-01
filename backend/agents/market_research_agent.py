import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

_SYSTEM = """You are a senior Market Research Analyst with deep expertise in product strategy and business intelligence.
Analyse sales data, product analytics, and market trends to uncover performance patterns and growth opportunities.

Focus on:
• Revenue and profit trends by product and region
• Best and worst performing categories
• Marketing spend efficiency (ROI)
• Seasonal or temporal patterns
• Untapped market segments and growth levers

Produce a structured, data-rich report. Cite specific numbers, percentages, and product names wherever possible."""


class MarketResearchAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Market Research Agent",
            description="Analyses sales data, product analytics, and market trends",
        )

    def run(self, state: dict) -> dict:
        ctx = self.retrieve(
            "sales revenue profit product performance region market growth analytics",
            k=10,
        )
        if not ctx:
            ctx = "No market or sales documents have been uploaded."

        prompt = f"""Analyse the following market and sales data to produce a **Market Research Summary**.

MARKET / SALES DATA:
{ctx}

Structure your report exactly as follows:

## 1. Market Overview
(Key top-line metrics: total revenue, units sold, profit margin)

## 2. Top Performing Products & Categories
(Best performers with revenue / units / rating figures)

## 3. Underperforming Areas
(Products, categories, or regions needing attention — with data)

## 4. Regional Performance Analysis
(Performance breakdown by geography)

## 5. Revenue & Profitability Trends
(Trends over time, seasonality, growth rates)

## 6. Marketing Spend Efficiency
(Return on marketing investment by product or category)

## 7. Growth Opportunities
(3–5 data-backed growth levers or untapped segments)

## 8. Key Risks
(Market risks, declining segments, dependency concerns)

## 9. Strategic Recommendations
(3–5 actionable recommendations backed by the data)

Include specific numbers wherever the data allows."""

        result = self.llm_call(_SYSTEM, prompt)
        state["market_research"] = result
        state["current_step"] = "Market Research Analysis complete"
        return state
