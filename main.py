import uuid
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel
from graph import build_graph, get_checkpointer_context, get_mcp_session_context, set_mcp_tools, init_audit_db
from schemas import ApprovalDecision, BIInsightReport

load_dotenv()

_app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with get_checkpointer_context() as checkpointer, get_mcp_session_context() as mcp_session:
        await checkpointer.setup()
        await init_audit_db()
        tools = await load_mcp_tools(mcp_session)
        set_mcp_tools(tools)
        _app_state["graph"] = build_graph(checkpointer)
        yield
    _app_state.clear()


app = FastAPI(title="Natural-Language BI Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    thread_id: str
    status: str
    report: BIInsightReport | None = None
    approval_request: dict | None = None


async def _to_response(thread_id: str, result: dict, config: dict, graph) -> QueryResponse:
    state_snapshot = await graph.aget_state(config)
    if state_snapshot.next:
        if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
            interrupt_val = state_snapshot.tasks[0].interrupts[0].value
            return QueryResponse(thread_id=thread_id, status="pending_approval", approval_request=interrupt_val)
    return QueryResponse(thread_id=thread_id, status="completed", report=result.get("report"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/queries", response_model=QueryResponse)
async def submit_query(request: QueryRequest) -> QueryResponse:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = _app_state["graph"]
    try:
        result = await graph.ainvoke({"question": request.question}, config=config)
        return await _to_response(thread_id, result, config, graph)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/queries/{thread_id}/resume", response_model=QueryResponse)
async def resume_query(thread_id: str, decision: ApprovalDecision) -> QueryResponse:
    config = {"configurable": {"thread_id": thread_id}}
    graph = _app_state["graph"]
    try:
        result = await graph.ainvoke(Command(resume=decision.model_dump()), config=config)
        return await _to_response(thread_id, result, config, graph)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)