"""FastAPI backend — AI Product Strategy Assistant."""
import os
import sys
from typing import List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OPENAI_API_KEY
from orchestrator.workflow import stream_analysis, run_chat
from utils.document_processor import DocumentProcessor
from utils.pdf_generator import generate_pdf_report
from vector_store.chroma_store import ChromaVectorStore

# ------------------------------------------------------------------ #
#  App setup                                                           #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="AI Product Strategy Assistant",
    description="Multi-agent AI system for product strategy analysis",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  Singletons                                                          #
# ------------------------------------------------------------------ #

processor = DocumentProcessor()
store = ChromaVectorStore()

# In-memory application state (use Redis/DB in production)
_state: dict = {
    "analysis_result": None,
    "analysis_running": False,
    "error": None,
}

# ------------------------------------------------------------------ #
#  Pydantic models                                                     #
# ------------------------------------------------------------------ #


class ChatRequest(BaseModel):
    query: str
    chat_history: List[dict] = []


class StatusResponse(BaseModel):
    running: bool
    completed: bool
    documents_loaded: int
    current_step: Optional[str] = None
    error: Optional[str] = None


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #


@app.get("/", tags=["Health"])
def root():
    return {"message": "AI Product Strategy Assistant API", "status": "running"}


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
def get_status():
    return StatusResponse(
        running=_state["analysis_running"],
        completed=_state["analysis_result"] is not None,
        documents_loaded=store.get_document_count(),
        current_step=(
            _state["analysis_result"].get("current_step")
            if _state["analysis_result"]
            else None
        ),
        error=_state["error"],
    )


@app.post("/api/upload", tags=["Data"])
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(default="general"),
):
    """Ingest a document into the ChromaDB vector store (embeddings are local — no API call)."""
    try:
        content = await file.read()
        chunks = processor.process_file(content, file.filename, doc_type)
        store.add_documents(chunks)
        return {
            "success": True,
            "filename": file.filename,
            "doc_type": doc_type,
            "chunks_created": len(chunks),
            "total_chunks": store.get_document_count(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/analyze", tags=["Analysis"])
def start_analysis(background_tasks: BackgroundTasks):
    """Kick off the 7-agent analysis pipeline in the background."""
    if _state["analysis_running"]:
        raise HTTPException(status_code=409, detail="Analysis is already running.")
    if not store.has_documents():
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload at least one document first.",
        )

    def _run():
        _state["analysis_running"] = True
        _state["error"] = None
        _state["analysis_result"] = None   # clear any previous run
        try:
            # stream_analysis() yields partial state after every agent node —
            # results appear in /api/results as each agent finishes
            for partial in stream_analysis():
                _state["analysis_result"] = partial
        except Exception as exc:
            _state["error"] = str(exc)
        finally:
            _state["analysis_running"] = False

    background_tasks.add_task(_run)
    return {"message": "Analysis started.", "status": "running"}


@app.get("/api/results", tags=["Analysis"])
def get_results():
    """
    Return analysis results — partial while running, full when done.
    'completed' is True whenever any agent has produced output, so the
    frontend can render sections incrementally rather than waiting for all 7.
    """
    r = _state["analysis_result"]
    if not r:
        return {"completed": False, "data": None}

    # completed = True as soon as the analysis pipeline has run at least once
    # (regardless of whether individual agents succeeded or failed).
    # The frontend uses this to switch from the "upload data" screen to the
    # results screen, where errors/missing sections are shown explicitly.
    return {
        "completed": not _state["analysis_running"],
        "still_running": _state["analysis_running"],
        "data": {
            "customer_insights": r.get("customer_insights"),
            "market_research": r.get("market_research"),
            "competitor_analysis": r.get("competitor_analysis"),
            "swot_analysis": r.get("swot_analysis"),
            "feature_priorities": r.get("feature_priorities"),
            "strategy_recommendations": r.get("strategy_recommendations"),
            "executive_summary": r.get("executive_summary"),
            "current_step": r.get("current_step"),
            "error": r.get("error"),
        },
    }


@app.post("/api/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """RAG-powered chat using uploaded documents and analysis results."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in .env")
    try:
        analysis = _state.get("analysis_result") or {}
        result = run_chat(request.query, request.chat_history, analysis)
        return {
            "response": result.get("chat_response", "No response generated."),
            "error": result.get("error"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/report/download", tags=["Report"])
def download_pdf():
    """Generate and stream the full strategy report as a PDF."""
    if not _state["analysis_result"]:
        raise HTTPException(status_code=404, detail="No analysis results available yet.")
    try:
        pdf_bytes = generate_pdf_report(_state["analysis_result"])
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=product_strategy_report.pdf"
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")


@app.delete("/api/reset", tags=["System"])
def reset():
    """Clear all documents and analysis results."""
    try:
        store.clear()
        _state["analysis_result"] = None
        _state["analysis_running"] = False
        _state["error"] = None
        return {"success": True, "message": "System reset successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
