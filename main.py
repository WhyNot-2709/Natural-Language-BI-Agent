import uuid
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from graph import build_graph, get_checkpointer_context, init_audit_db
from schemas import ApprovalDecision, BIInsightReport

load_dotenv()

_app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with get_checkpointer_context() as checkpointer:
        await checkpointer.setup()
        await init_audit_db()
        _app_state["graph"] = build_graph(checkpointer)
        yield
    _app_state.clear()

app = FastAPI(title="Natural-language BI Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    thread_id: str
    status: str
    report: BIInsightReport | None = None
    approval_request: dict | None = None


async def _to_response(thread_id: str, result: dict, config: dict, graph) -> QueryResponse:
    # Modern LangGraph 1.2+ interrupt checking using the state snapshot
    state_snapshot = await graph.aget_state(config)
    
    # If there is a 'next' node, the graph is paused
    if state_snapshot.next:
        # Extract the interrupt payload from the pending task
        if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
            interrupt_val = state_snapshot.tasks[0].interrupts[0].value
            return QueryResponse(
                thread_id=thread_id, 
                status="pending_approval", 
                approval_request=interrupt_val
            )
            
    # If no next node, it finished completely
    return QueryResponse(
        thread_id=thread_id, 
        status="completed", 
        report=result.get("report")
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/queries", response_model=QueryResponse)
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
