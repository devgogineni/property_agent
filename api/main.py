import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

import config
from db.connection import get_connection
from db.dashboard_metrics import fetch_conversation_series, fetch_dashboard_summary
from db.feedback import init_feedback_table, insert_feedback
from db.llm_call_metrics import (
    fetch_llm_call_metrics,
    init_llm_call_metrics_table,
    insert_llm_call_metric,
)
from db.prompt_log import fetch_prompt_logs, init_prompt_log_table, insert_prompt_log
from rag.agent import PropertyRAGAgent
from rag.resolver import LocationResolver
from rag.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationMetricPoint,
    DashboardSummary,
    FeedbackRequest,
    LLMCallMetricEntry,
    PromptLogEntry,
)
from utils.embedder import Embedder

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 30000")

    embedder = Embedder(path=config.EMBEDDING_MODEL_PATH)
    resolver = LocationResolver.load(config.LOCATION_INDEX_PATH, embedder)
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    log_conn = get_connection(config.PROMPT_LOG_DB_PATH, check_same_thread=False)
    init_prompt_log_table(log_conn)
    init_llm_call_metrics_table(log_conn)
    init_feedback_table(log_conn)

    app.state.conn = conn
    app.state.resolver = resolver
    app.state.client = client
    app.state.log_conn = log_conn

    yield

    conn.close()
    log_conn.close()


app = FastAPI(title="Estate Agent RAG API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_agent() -> PropertyRAGAgent:
    return PropertyRAGAgent(
        conn=app.state.conn,
        llm_client=app.state.client,
        resolver=app.state.resolver,
        model=config.CHAT_MODEL,
        extraction_model=config.EXTRACTION_MODEL,
    )


@app.get("/health")
def health():
    ppd_rows = app.state.conn.execute("SELECT COUNT(*) FROM ppd").fetchone()[0]
    hpi_rows = app.state.conn.execute("SELECT COUNT(*) FROM hpi").fetchone()[0]
    return {"status": "ok", "ppd_rows": ppd_rows, "hpi_rows": hpi_rows}


@app.get("/")
def index_page():
    return FileResponse(str(STATIC_DIR / "chat.html"))


@app.get("/prompts")
def prompts_page():
    return FileResponse(str(STATIC_DIR / "prompts.html"))


@app.get("/dashboard")
def dashboard_page():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    agent = get_agent()
    answer = agent.rag(payload.question)
    results = agent.last_search_results or {}
    total_cost = agent.total_cost()

    input_tokens = sum(usage.input_tokens for usage in agent.usages)
    output_tokens = sum(usage.output_tokens for usage in agent.usages)
    request_id = insert_prompt_log(
        app.state.log_conn,
        question=payload.question,
        answer=answer,
        model=agent.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=total_cost,
    )
    for call in agent.calls:
        insert_llm_call_metric(app.state.log_conn, request_id=str(request_id), call=call)

    # LLM-as-a-judge: inline relevance check, run after total_cost_usd is
    # computed so the judge's own call doesn't inflate the answer-serving
    # cost shown to the user/dashboard - its cost is still fully visible via
    # llm_call_metrics (call_type="judge") for anyone drilling into a
    # conversation's calls.
    verdict, judge_call = agent.judge_relevance(payload.question, answer)
    insert_llm_call_metric(app.state.log_conn, request_id=str(request_id), call=judge_call)
    insert_feedback(
        app.state.log_conn,
        request_id=str(request_id),
        source="judge",
        relevance=verdict.relevance,
        explanation=verdict.explanation,
    )

    return ChatResponse(
        request_id=request_id,
        answer=answer,
        extracted_query=agent.last_extracted,
        resolved_location=agent.last_resolved,
        num_ppd_matches=len(results.get("ppd_rows", [])),
        num_hpi_matches=len(results.get("hpi_rows", [])),
        total_cost_usd=total_cost,
        relevance=verdict.relevance,
        relevance_explanation=verdict.explanation,
    )


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest) -> dict:
    exists = app.state.log_conn.execute(
        "SELECT 1 FROM prompt_logs WHERE id = ?", (payload.request_id,)
    ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="Unknown request_id")

    insert_feedback(
        app.state.log_conn,
        request_id=str(payload.request_id),
        score=payload.score,
        source="user",
    )
    return {"status": "ok"}


@app.get("/api/prompt-logs", response_model=list[PromptLogEntry])
def prompt_logs(limit: int = 200) -> list[PromptLogEntry]:
    rows = fetch_prompt_logs(app.state.log_conn, limit=limit)
    return [PromptLogEntry(**dict(row)) for row in rows]


@app.get("/api/llm-call-metrics", response_model=list[LLMCallMetricEntry])
def llm_call_metrics(request_id: str | None = None, limit: int = 200) -> list[LLMCallMetricEntry]:
    rows = fetch_llm_call_metrics(app.state.log_conn, request_id=request_id, limit=limit)
    return [LLMCallMetricEntry(**dict(row)) for row in rows]


@app.get("/api/dashboard-summary", response_model=DashboardSummary)
def dashboard_summary() -> DashboardSummary:
    return DashboardSummary(**fetch_dashboard_summary(app.state.log_conn))


@app.get("/api/dashboard-conversations", response_model=list[ConversationMetricPoint])
def dashboard_conversations(limit: int = 100) -> list[ConversationMetricPoint]:
    rows = fetch_conversation_series(app.state.log_conn, limit=limit)
    return [ConversationMetricPoint(**dict(row)) for row in rows]
