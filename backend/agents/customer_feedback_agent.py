import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

_SYSTEM = """You are an expert Customer Insights Analyst specialising in product strategy.
Analyse customer feedback, reviews, ratings, and survey data to surface actionable insights.

Focus on:
• Sentiment distribution and trends
• Recurring pain points and complaints
• Most-requested features and improvements
• Customer satisfaction drivers
• At-risk customer signals

Produce a clear, structured report. Use headings, bullet points, and cite data patterns.
Be specific — reference product names, rating scores, and review patterns where visible."""


class CustomerFeedbackAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Customer Feedback Agent",
            description="Analyses customer reviews, ratings, and survey responses",
        )

    def run(self, state: dict) -> dict:
        ctx = self.retrieve(
            "customer feedback reviews ratings satisfaction pain points feature requests",
            k=10,
        )
        if not ctx:
            ctx = "No customer feedback documents have been uploaded."

        prompt = f"""Analyse the following customer data and produce a comprehensive **Customer Insights Report**.

CUSTOMER DATA:
{ctx}

Structure your report exactly as follows:

## 1. Executive Overview
(2–3 sentence summary of overall customer health)

## 2. Sentiment Analysis
(Overall positive / neutral / negative breakdown with evidence)

## 3. Top Pain Points
(Top 5 complaints with frequency signals)

## 4. Most-Requested Features / Improvements
(Top 5 requests with supporting quotes or patterns)

## 5. Customer Satisfaction Drivers
(What customers love — key differentiators)

## 6. At-Risk Customer Signals
(Warning signs of dissatisfaction or churn)

## 7. Recommendations
(3–5 specific, actionable recommendations for the product team)

Be precise and data-backed. Reference product names, categories, or regions when visible."""

        result = self.llm_call(_SYSTEM, prompt)
        state["customer_insights"] = result
        state["current_step"] = "Customer Feedback Analysis complete"
        return state
