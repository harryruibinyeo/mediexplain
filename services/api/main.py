"""
MediExplain SG — API Service

FastAPI application that:
  1. Accepts a PDF upload at POST /explain
  2. Extracts text from the PDF using pdfplumber
  3. Runs the LangChain agent (Qwen 2.5 via vLLM + pgvector search)
  4. Returns a plain-language explanation with citations

Observability:
  - Structured JSON logs via structlog on every request
  - Prometheus metrics at GET /metrics (auto-instrumented + custom agent metrics)
    Scraped by Prometheus every 15s, visualised in Grafana

Environment variables:
  DATABASE_URL  — PostgreSQL with pgvector
  VLLM_URL      — base URL of vLLM server (e.g. http://inference:8000)
  MODEL_NAME    — model name served by vLLM
  EMBED_MODEL   — sentence-transformers model (must match ingest service)
"""

import os
import tempfile
from contextlib import asynccontextmanager

import pdfplumber
import psycopg2
import structlog
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pgvector.psycopg2 import register_vector
from prometheus_fastapi_instrumentator import Instrumentator
from sentence_transformers import SentenceTransformer

import agent
import chat

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ["DATABASE_URL"]
VLLM_URL     = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-AWQ")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "all-mpnet-base-v2")


# ---------------------------------------------------------------------------
# App lifecycle — load models and connections once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api_starting", vllm_url=VLLM_URL, model=MODEL_NAME, embed_model=EMBED_MODEL)

    # Load embedding model (same model used by the ingest service)
    log.info("loading_embed_model", model=EMBED_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)
    log.info("embed_model_loaded", model=EMBED_MODEL,
             dim=embed_model.get_sentence_embedding_dimension())

    # Connect to PostgreSQL + pgvector
    db_display = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    log.info("connecting_db", host=db_display)
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    log.info("db_connected")

    log.info("agent_ready", tools=["search_conditions", "search_medications"])

    # Store on app.state so route handlers can access them
    app.state.conn        = conn
    app.state.embed_model = embed_model

    log.info("api_ready")
    yield

    conn.close()
    log.info("api_shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MediExplain SG API",
    description="Explains medical documents in plain language using RAG + LangChain",
    lifespan=lifespan,
)

# Auto-instrument all FastAPI endpoints — adds /metrics with:
#   http_requests_total, http_request_duration_seconds, http_requests_in_progress
Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check — used by Kubernetes liveness probe."""
    return {"status": "ok"}


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """
    Accept a PDF medical document and return a plain-language explanation.

    The LangChain agent searches the HealthHub knowledge base (pgvector)
    to ground its explanation in Singapore-specific medical information.
    """
    log.info("explain_request", filename=file.filename, content_type=file.content_type)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read and extract PDF text
    content = await file.read()
    log.info("pdf_received", filename=file.filename, size_bytes=len(content))

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        with pdfplumber.open(tmp.name) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]

    doc_text = "\n\n".join(pages_text).strip()
    log.info("pdf_extracted", filename=file.filename, pages=len(pages_text), chars=len(doc_text))

    if len(doc_text) < 50:
        raise HTTPException(status_code=422,
                            detail="Could not extract readable text from this PDF.")

    # Truncate to fit within the 2048-token context window (RTX 3070 8GB VRAM)
    # ~1500 chars ≈ 350 tokens, leaving room for system prompt + tools + output
    if len(doc_text) > 1800:
        doc_text = doc_text[:1800] + "\n[Document truncated for length]"
        log.info("pdf_truncated", filename=file.filename)

    result = await agent.run(
        doc_text=doc_text,
        conn=app.state.conn,
        embed_model=app.state.embed_model,
        vllm_url=VLLM_URL,
        model_name=MODEL_NAME,
    )

    log.info(
        "explain_complete",
        filename=file.filename,
        tool_calls=result["meta"]["tool_calls"],
        citations=len(result["citations"]),
        latency_s=result["meta"]["latency_s"],
    )

    return JSONResponse(result)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Accept a plain-English question and return an answer grounded in the
    knowledge base. Uses a 4-step NL2SQL pipeline:
      1. Qwen generates a SELECT query from the question
      2. Python validates the SQL (blocks destructive operations)
      3. psycopg2 executes the query against knowledge_base
      4. Qwen explains the results in plain English
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    log.info("chat_request", question=request.question[:80])

    result = await chat.run(
        question=request.question,
        conn=app.state.conn,
        vllm_url=VLLM_URL,
        model_name=MODEL_NAME,
    )

    log.info(
        "chat_complete",
        row_count=result["meta"]["row_count"],
        latency_s=result["meta"]["latency_s"],
    )

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
