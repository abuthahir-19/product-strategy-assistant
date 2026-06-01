"""Streamlit frontend — AI Product Strategy Assistant."""
import os
import time
from pathlib import Path

import requests
import streamlit as st

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="AI Product Strategy Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Read from environment so the same codebase works locally and on Render
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ------------------------------------------------------------------ #
#  CSS                                                                 #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
/* Header */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
    color: white;
    padding: 22px 30px;
    border-radius: 12px;
    margin-bottom: 24px;
    text-align: center;
}
.app-header h1 { margin: 0; font-size: 2rem; }
.app-header p  { margin: 4px 0 0; opacity: .85; font-size: .95rem; }

/* Agent status badges */
.badge-done    { color: #27ae60; font-weight: 600; }
.badge-running { color: #e67e22; font-weight: 600; }
.badge-pending { color: #aaa; }

/* Insight cards */
.insight-card {
    background: #f8faff;
    border-left: 4px solid #2e6da4;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

/* Metrics row */
.metric-box {
    background: white;
    border: 1px solid #dde3ee;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}

/* Chat bubbles */
.chat-user {
    background: #e3f0ff;
    border-radius: 12px 12px 4px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
}
.chat-bot {
    background: #f3f3f3;
    border-radius: 12px 12px 12px 4px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 80%;
}

div[data-testid="stMarkdownContainer"] h2 { color: #1e3a5f; }
div[data-testid="stMarkdownContainer"] h3 { color: #2e6da4; }
.stButton > button { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Session state defaults                                              #
# ------------------------------------------------------------------ #
for _k, _v in {
    "chat_history": [],
    "uploaded_files": [],
    "poll_analysis": False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ------------------------------------------------------------------ #
#  API helpers                                                         #
# ------------------------------------------------------------------ #

def _get(path: str, **kwargs):
    try:
        return requests.get(f"{BACKEND_URL}{path}", timeout=10, **kwargs)
    except Exception:
        return None


def _post(path: str, **kwargs):
    try:
        return requests.post(f"{BACKEND_URL}{path}", timeout=120, **kwargs)
    except Exception:
        return None


def _delete(path: str):
    try:
        return requests.delete(f"{BACKEND_URL}{path}", timeout=10)
    except Exception:
        return None


def backend_alive() -> bool:
    """Try up to 3 times with increasing timeouts to handle Render cold starts."""
    for timeout in (5, 10, 20):
        try:
            r = requests.get(f"{BACKEND_URL}/", timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


def get_status() -> dict:
    r = _get("/api/status")
    return r.json() if (r and r.status_code == 200) else {}


def upload_file(file, doc_type: str) -> dict:
    r = requests.post(
        f"{BACKEND_URL}/api/upload",
        files={"file": (file.name, file.getvalue(), "application/octet-stream")},
        data={"doc_type": doc_type},
        timeout=60,
    )
    return r.json() if r.status_code == 200 else {"error": r.text}


def start_analysis() -> dict:
    r = _post("/api/analyze")
    return r.json() if (r and r.status_code == 200) else {"error": getattr(r, "text", "Request failed")}


def get_results() -> dict:
    r = _get("/api/results")
    return r.json() if (r and r.status_code == 200) else {}


def chat_query(query: str, history: list) -> dict:
    r = _post("/api/chat", json={"query": query, "chat_history": history})
    return r.json() if (r and r.status_code == 200) else {"error": getattr(r, "text", "Request failed")}


def download_pdf() -> bytes | None:
    r = _get("/api/report/download")
    return r.content if (r and r.status_code == 200) else None


def reset_system() -> dict:
    r = _delete("/api/reset")
    return r.json() if (r and r.status_code == 200) else {"error": "Reset failed"}

# ------------------------------------------------------------------ #
#  Header                                                              #
# ------------------------------------------------------------------ #
st.markdown("""
<div class="app-header">
    <h1>🤖 AI-Powered Product Strategy Assistant</h1>
    <p>Multi-Agent System &nbsp;|&nbsp; LangGraph &nbsp;|&nbsp; RAG &nbsp;|&nbsp; GPT-4o Mini</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Backend health check                                                #
# ------------------------------------------------------------------ #
if not backend_alive():
    is_local = "localhost" in BACKEND_URL or "127.0.0.1" in BACKEND_URL
    if is_local:
        st.error(
            "**Backend server is not running.**  \n\n"
            "Open a terminal and run:\n"
            "```\ncd backend\npython main.py\n```"
        )
    else:
        st.error(
            f"**Cannot reach the backend API.**  \n\n"
            f"Configured URL: `{BACKEND_URL}`  \n\n"
            "**Check the following in your Render dashboard:**\n"
            "1. The **backend service** (`product-strategy-api`) is deployed and shows **Live**\n"
            "2. The **frontend service** has `BACKEND_URL` set to the backend's Render URL\n"
            "3. The backend URL should be `https://your-service-name.onrender.com` (no port number)\n\n"
            "The backend may also be waking up from idle — wait 30 seconds and refresh."
        )
    st.stop()

# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## 📁 Upload Data")

    DOC_TYPES = {
        "Customer Reviews / Feedback": "customer_feedback",
        "Sales Data / Analytics":      "sales_data",
        "Market Research":             "market_research",
        "Competitor Information":      "competitor_info",
        "Feature Requests":            "feature_requests",
        "Survey Responses":            "survey_data",
        "General / Other":             "general",
    }

    label = st.selectbox("Document type", list(DOC_TYPES.keys()))
    doc_type = DOC_TYPES[label]

    uploaded = st.file_uploader(
        "Choose file(s)",
        type=["pdf", "csv", "txt", "docx", "md"],
        accept_multiple_files=True,
        help="Supported: PDF, CSV, TXT, DOCX, MD",
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.uploaded_files:
                with st.spinner(f"Ingesting {f.name} …"):
                    res = upload_file(f, doc_type)
                if "error" not in res:
                    st.success(f"✅ {f.name} — {res.get('chunks_created', 0)} chunks")
                    st.session_state.uploaded_files.append(f.name)
                else:
                    st.error(f"❌ {res['error']}")

    # ---- Status ---- #
    st.markdown("---")
    status = get_status()
    docs_n = status.get("documents_loaded", 0)

    col_a, col_b = st.columns(2)
    col_a.metric("Chunks loaded", docs_n)
    if status.get("completed"):
        col_b.success("Done ✅")
    elif status.get("running"):
        col_b.warning("Running ⏳")
    else:
        col_b.info("Ready")

    # ---- Run Analysis ---- #
    st.markdown("---")
    st.markdown("### 🚀 Run Analysis")

    if st.button("▶ Start Multi-Agent Analysis", use_container_width=True):
        res = start_analysis()
        if "error" not in res or not res.get("error"):
            st.success("Analysis started!")
            st.session_state.poll_analysis = True
            st.rerun()
        else:
            st.error(res.get("error"))

    if st.button("🔄 Reset System", use_container_width=True):
        res = reset_system()
        if "success" in res:
            st.session_state.uploaded_files = []
            st.session_state.chat_history = []
            st.session_state.poll_analysis = False
            st.success("Reset complete.")
            st.rerun()
        else:
            st.error(res.get("error", "Reset failed"))

    # ---- Agent pipeline ---- #
    st.markdown("---")
    st.markdown("### 🤖 Agent Pipeline")

    # Phase 1 groups 3 agents that run in parallel
    PIPELINE = [
        ("Phase 1 ⚡ Parallel",
         "Customer Feedback · Market Research · Competitor Analysis",
         ["customer_insights", "market_research", "competitor_analysis"]),
        ("Phase 2",  "SWOT Analysis Agent",               ["swot_analysis"]),
        ("Phase 3",  "Feature Prioritization Agent",       ["feature_priorities"]),
        ("Phase 4",  "Strategy Recommendation Agent",      ["strategy_recommendations"]),
        ("Phase 5",  "Executive Report Agent",             ["executive_summary"]),
    ]

    results_now = get_results()
    completed_keys = set()
    if results_now.get("completed") and results_now.get("data"):
        d = results_now["data"]
        for _, _, keys in PIPELINE:
            for k in keys:
                if d.get(k):
                    completed_keys.add(k)

    for i, (phase, label, keys) in enumerate(PIPELINE, 1):
        phase_done = all(k in completed_keys for k in keys)
        if phase_done:
            st.markdown(
                f'<span class="badge-done">✅ {phase}: {label}</span>',
                unsafe_allow_html=True,
            )
        elif status.get("running"):
            st.markdown(
                f'<span class="badge-running">⏳ {phase}: {label}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="badge-pending">⬜ {phase}: {label}</span>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------ #
#  Auto-poll while running                                             #
# ------------------------------------------------------------------ #
status = get_status()
if status.get("running"):
    step = status.get("current_step", "Processing…")
    with st.spinner(f"Analysis in progress — {step}"):
        time.sleep(4)
    st.rerun()
elif st.session_state.poll_analysis and status.get("completed"):
    st.session_state.poll_analysis = False

# ------------------------------------------------------------------ #
#  Main tabs                                                           #
# ------------------------------------------------------------------ #
tab_chat, tab_analysis, tab_report = st.tabs([
    "💬 Chat Assistant",
    "📊 Analysis Results",
    "📄 Report & Download",
])

# ================================================================== #
#  TAB 1 — CHAT                                                       #
# ================================================================== #
with tab_chat:
    st.markdown("## 💬 Product Strategy Chat")
    st.caption("Ask questions about your data, analysis results, or any product strategy topic.")

    # Display history
    if not st.session_state.chat_history:
        st.info(
            "👋 **Get started:** Upload your documents, then ask me anything.  \n"
            "Example: *'What are the top customer complaints?'* or *'Which product should we invest in next?'*"
        )
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about your product strategy…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Thinking…"):
            resp = chat_query(prompt, st.session_state.chat_history[:-1])
        bot_reply = resp.get("response", "Unable to generate a response.")
        if resp.get("error"):
            bot_reply = f"⚠️ {resp['error']}"
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        st.rerun()

    if st.session_state.chat_history:
        st.button("🗑️ Clear chat", on_click=lambda: st.session_state.update(chat_history=[]))

    # Suggested questions
    with st.expander("💡 Suggested questions"):
        suggs = [
            "What are the top 3 customer pain points?",
            "Which products have the highest profit margin?",
            "What features should we build next?",
            "What does the SWOT analysis say about our growth opportunities?",
            "What is the recommended 90-day action plan?",
            "Which region is underperforming and why?",
            "How should we position against competitors?",
        ]
        for s in suggs:
            if st.button(s, key=f"sugg_{s}"):
                st.session_state.chat_history.append({"role": "user", "content": s})
                with st.spinner("Thinking…"):
                    resp = chat_query(s, st.session_state.chat_history[:-1])
                bot_reply = resp.get("response", "Unable to generate a response.")
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                st.rerun()

# ================================================================== #
#  TAB 2 — ANALYSIS RESULTS                                           #
# ================================================================== #
with tab_analysis:
    st.markdown("## 📊 Analysis Results")

    results = get_results()

    if not results.get("completed"):
        st.info("Upload documents and click **▶ Start Multi-Agent Analysis** in the sidebar.")

        st.markdown("### How it works")
        cols = st.columns(4)
        steps = [
            ("📁", "Upload Data", "CSV, PDF, TXT, DOCX — customer reviews, sales reports, market research"),
            ("⚙️", "Multi-Agent Processing", "7 specialised AI agents analyse different aspects of your data"),
            ("💡", "Insight Generation", "Agents share insights and synthesise a complete strategy picture"),
            ("📄", "Report Creation", "Download a professional PDF strategy report"),
        ]
        for col, (icon, title, desc) in zip(cols, steps):
            with col:
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)
    else:
        data = results["data"]

        if data.get("error"):
            st.error(f"Analysis error: {data['error']}")

        # Summary metrics
        completed_count = sum(1 for _, k in AGENTS if data.get(k))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Agents Completed", f"{completed_count}/7")
        m2.metric("Chunks Analysed", store_docs := get_status().get("documents_loaded", 0))
        m3.metric("Report Sections", completed_count)
        m4.metric("Status", "✅ Complete" if completed_count == 7 else "⚡ Partial")

        st.markdown("---")

        # Display each section
        SECTION_DEFS = [
            ("📋 Executive Summary",                  "executive_summary",         True),
            ("👥 Customer Insights Report",           "customer_insights",         False),
            ("📈 Market Research Summary",            "market_research",           False),
            ("🏆 Competitor Analysis",                "competitor_analysis",        False),
            ("🎯 SWOT Analysis",                      "swot_analysis",             False),
            ("⭐ Feature Prioritization",             "feature_priorities",         False),
            ("🚀 Strategic Recommendations",          "strategy_recommendations",  False),
        ]

        for title, key, expanded in SECTION_DEFS:
            content = data.get(key, "")
            if content:
                with st.expander(title, expanded=expanded):
                    st.markdown(content)
            else:
                with st.expander(f"{title} *(not available)*", expanded=False):
                    st.caption("This section was not generated. Re-run the analysis after uploading relevant documents.")

# ================================================================== #
#  TAB 3 — REPORT & DOWNLOAD                                          #
# ================================================================== #
with tab_report:
    st.markdown("## 📄 Report & Download")

    results = get_results()

    left, right = st.columns([3, 2])

    with left:
        if results.get("completed"):
            st.success("✅ Analysis complete — your report is ready.")

            pdf = download_pdf()
            if pdf:
                st.download_button(
                    label="⬇️ Download PDF Strategy Report",
                    data=pdf,
                    file_name="product_strategy_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.warning("PDF generation is unavailable. Check that `reportlab` is installed.")

            st.markdown("### Report Contents")
            for idx, (title, key, _) in enumerate(SECTION_DEFS, 1):
                icon = "✅" if results["data"].get(key) else "⬜"
                st.markdown(f"{icon} **{idx}.** {title.split(' ', 1)[1]}")
        else:
            st.info("Run the analysis first to generate a downloadable report.")
            st.markdown("""
**The PDF report includes:**
1. Executive Summary
2. Customer Insights Report
3. Market Research Summary
4. Competitor Analysis Report
5. SWOT Analysis
6. Feature Prioritization Recommendations
7. Strategic Recommendations & 90-day Action Plan
            """)

    with right:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
```
┌─────────────────────────────┐
│      Data Ingestion         │
│  CSV · PDF · TXT · DOCX     │
└────────────┬────────────────┘
             │ embeddings
             ▼
┌─────────────────────────────┐
│   ChromaDB Vector Store     │
│  (text-embedding-3-small)   │
└────────────┬────────────────┘
             │ RAG retrieval
             ▼
┌─────────────────────────────┐
│   LangGraph Orchestrator    │
├─────────────────────────────┤
│ 1. Customer Feedback Agent  │
│ 2. Market Research Agent    │
│ 3. Competitor Analysis Agent│
│ 4. SWOT Analysis Agent      │
│ 5. Feature Priority Agent   │
│ 6. Strategy Rec. Agent      │
│ 7. Executive Report Agent   │
└────────────┬────────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
 PDF Report     Chat (RAG)
```
        """)

        st.markdown("### 🛠️ Tech Stack")
        stack = {
            "LLM": "GPT-4o Mini",
            "Embeddings": "text-embedding-3-small",
            "Agents": "LangGraph + LangChain",
            "Vector DB": "ChromaDB",
            "Backend": "FastAPI",
            "Frontend": "Streamlit",
        }
        for tech, val in stack.items():
            st.markdown(f"- **{tech}:** {val}")
