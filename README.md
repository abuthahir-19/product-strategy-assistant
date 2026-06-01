# AI-Powered Product Strategy Assistant

A multi-agent AI system that helps Product Managers transform business data into
actionable strategic insights.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                            │
│         CSV · PDF · TXT · DOCX  →  ChromaDB Vector Store        │
└──────────────────────────────────────┬───────────────────────────┘
                                       │  RAG retrieval
                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│               LANGGRAPH MULTI-AGENT PIPELINE                     │
│                                                                  │
│  [1] Customer Feedback Agent     — sentiment, pain points        │
│       ↓                                                          │
│  [2] Market Research Agent       — sales trends, performance     │
│       ↓                                                          │
│  [3] Competitor Analysis Agent   — competitive landscape         │
│       ↓                                                          │
│  [4] SWOT Analysis Agent         — synthesises all 3 above      │
│       ↓                                                          │
│  [5] Feature Prioritization Agent — RICE/MoSCoW framework       │
│       ↓                                                          │
│  [6] Strategy Recommendation Agent — action plan + roadmap      │
│       ↓                                                          │
│  [7] Executive Report Agent      — C-suite executive summary    │
└──────────────────────────────────────┬───────────────────────────┘
                                       │
                          ┌────────────┴──────────────┐
                          ▼                           ▼
                    PDF Report               Chat (RAG Q&A)
```

## Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| LLM          | GPT-4o Mini (OpenAI)                |
| Embeddings   | text-embedding-3-small (OpenAI)     |
| Agents       | LangGraph + LangChain               |
| Vector DB    | ChromaDB                            |
| Backend      | FastAPI + Uvicorn                   |
| Frontend     | Streamlit                           |
| PDF Report   | ReportLab                           |

## Quick Start

### 1. Prerequisites

- Python 3.10+
- An OpenAI API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Windows
copy .env.example .env
notepad .env          # add your OPENAI_API_KEY

# Mac/Linux
cp .env.example .env
nano .env
```

`.env` contents:
```
OPENAI_API_KEY=sk-...your-key-here...
```

### 4. Start the app

**Windows (one-click):**
```
double-click start.bat
```

**Manual (two terminals):**

Terminal 1 — Backend:
```bash
cd backend
python main.py
# → http://localhost:8000
```

Terminal 2 — Frontend:
```bash
cd frontend
streamlit run app.py
# → http://localhost:8501
```

### 5. Use the app

1. Open **http://localhost:8501** in your browser
2. Upload documents via the sidebar (CSV, PDF, TXT, DOCX)
   - The included `Sample Sales Data.csv` is a great starting point
3. Click **▶ Start Multi-Agent Analysis**
4. Watch 7 agents process your data sequentially
5. Explore results in the **Analysis Results** tab
6. Chat with your data in the **Chat Assistant** tab
7. Download the PDF report from the **Report & Download** tab

## API Reference

The FastAPI backend exposes these endpoints:

| Method | Path                    | Description                          |
|--------|-------------------------|--------------------------------------|
| GET    | `/`                     | Health check                         |
| GET    | `/api/status`           | System / analysis status             |
| POST   | `/api/upload`           | Upload a document                    |
| POST   | `/api/analyze`          | Start the multi-agent analysis       |
| GET    | `/api/results`          | Retrieve analysis results            |
| POST   | `/api/chat`             | RAG-powered chat query               |
| GET    | `/api/report/download`  | Download PDF strategy report         |
| DELETE | `/api/reset`            | Clear documents and analysis         |

Interactive docs: **http://localhost:8000/docs**

## Supported Input Formats

| Format | Use case                                        |
|--------|-------------------------------------------------|
| CSV    | Sales data, product analytics, feature requests |
| PDF    | Market research, competitor reports             |
| TXT    | Customer reviews, survey responses              |
| DOCX   | Any business document                           |
| MD     | Internal notes, product specs                  |

## Expected Outputs

- **Customer Insights Report** — sentiment, pain points, feature requests
- **Market Research Summary** — performance trends, growth opportunities
- **Competitor Analysis** — landscape, gaps, differentiation strategies
- **SWOT Analysis** — with SO/ST/WO/WT strategic implications
- **Feature Prioritization** — RICE/MoSCoW tiers, quick wins
- **Strategic Action Plan** — 30/60/90 day roadmap, KPIs
- **Executive Summary** — C-suite ready, decision-focused
- **Downloadable PDF** — professional formatted report

## Project Structure

```
ProductStrategistAssistant/
├── backend/
│   ├── main.py                          # FastAPI server
│   ├── config.py                        # Environment config
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── customer_feedback_agent.py
│   │   ├── market_research_agent.py
│   │   ├── competitor_analysis_agent.py
│   │   ├── swot_analysis_agent.py
│   │   ├── feature_prioritization_agent.py
│   │   ├── strategy_recommendation_agent.py
│   │   └── executive_report_agent.py
│   ├── orchestrator/
│   │   └── workflow.py                  # LangGraph state machine
│   ├── vector_store/
│   │   └── chroma_store.py              # ChromaDB wrapper
│   └── utils/
│       ├── document_processor.py        # File ingestion + chunking
│       └── pdf_generator.py             # ReportLab PDF builder
├── frontend/
│   └── app.py                           # Streamlit UI
├── data/                                # Sample data directory
├── .env.example
├── requirements.txt
├── start.bat                            # Windows one-click launcher
└── README.md
```

## Evaluation Criteria Coverage

| Criterion                         | Implementation                              |
|-----------------------------------|---------------------------------------------|
| Successful Deployment (30%)       | FastAPI + Streamlit, one-click start.bat    |
| Quality of AI Insights (35%)      | 7 specialised agents with rich prompts      |
| Multi-Agent Design & UX (35%)     | LangGraph pipeline, clean Streamlit UI      |

## Bonus Features Implemented

- ✅ Advanced multi-agent collaboration (agents share outputs)
- ✅ Feature opportunity scoring (RICE + MoSCoW)
- ✅ Product roadmap generation (Q1–Q4 suggestions)
- ✅ Downloadable PDF report (ReportLab)
- ✅ Interactive chat with RAG
- ✅ 30/60/90 day action plan
