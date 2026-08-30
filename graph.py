import os
import json
import sqlite3
import chromadb
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import interrupt
from langgraph.graph import END, START, StateGraph

# NEW: Postgres Checkpointer Imports
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg

# UPDATED: Removed AUDIT_DB_PATH and CHECKPOINT_DB_PATH
from dataset import BUSINESS_DB_PATH, CHROMA_DB_DIR

from schemas import (
    BIInsightReport,
    ColumnInfo,
    GoldenQueryExample,
    IntakeClassification,
    QueryPlan,
    RetrievedContext,
    ValidationResult,
    TableSchema,
    ResolvedEntity,
    ApprovalDecision,
    ApprovalRequest
)

load_dotenv()

# NEW: Postgres Database URL
DATABASE_URL = os.environ.get("DATABASE_URL")

SCHEMA_TOP_K = int(os.environ.get("SCHEMA_TOP_K", "5"))
GOLDEN_QUERY_TOP_K = int(os.environ.get("GOLDEN_QUERY_TOP_K", 3))
ENTITY_TOP_K = int(os.environ.get("ENTITY_TOP_K", 10))
ENTITY_MAX_DISTANCE = float(os.environ.get("ENTITY_MAX_DISTANCE", "0.35"))
GLOSSARY_TOP_K = int(os.environ.get("GLOSSARY_TOP_K", "5"))
GLOSSARY_MAX_DISTANCE = float(os.environ.get("GLOSSARY_MAX_DISTANCE", "0.35"))

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_LITE_MODEL = os.environ.get("GEMINI_LITE_MODEL", "gemini-3.6-flash-lite")
GROQ_FAILOVER_MODEL = os.environ.get("GROQ_FAILOVER_MODEL", "openai/gpt-oss-120b")

SQL_GEN_MAX_RETRIES = int(os.environ.get("SQL_GEN_MAX_RETRIES", "3"))
HIGH_ROW_COUNT_THRESHOLD = int(os.environ.get("HIGH_ROW_COUNT_THRESHOLD", "500"))

_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
_embedder = GoogleGenerativeAIEmbeddings(model=os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"))
_gemini_flash = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
_gemini_lite = ChatGoogleGenerativeAI(model=GEMINI_LITE_MODEL)
_groq_failover = ChatGroq(model=GROQ_FAILOVER_MODEL)

class FailoverRunnable:
    def __init__(self, primary, failover, schema):
        self._primary = primary
        self._failover = failover
        self._schema = schema

    async def ainvoke(self, prompt, **kwargs):
        try:
            return await self._primary.ainvoke(prompt, **kwargs)
        except Exception as primary_error:
            try:
                # Inject the exact JSON schema rules into the prompt
                schema_rules = json.dumps(self._schema.model_json_schema())
                failover_prompt = f"{prompt}\n\nRespond ONLY with valid JSON that strictly matches this exact schema structure:\n{schema_rules}"
                
                return await self._failover.ainvoke(failover_prompt, **kwargs)
            except Exception as failover_error:
                raise RuntimeError(
                    f"Both Gemini and Groq failed. Gemini: {primary_error}, Groq: {failover_error}"
                ) from failover_error

def get_structured_llm(schema: type, lite: bool = False) -> FailoverRunnable:
    primary = (_gemini_lite if lite else _gemini_flash).with_structured_output(schema)
    failover = _groq_failover.with_structured_output(schema, method="json_mode")
    return FailoverRunnable(primary, failover, schema)

_mcp_client = MultiServerMCPClient(
    {
        "business_db_catalog": {
            "command": "python",
            "args": ["mcp_server.py"],
            "transport": "stdio"
        }
    }
)
_mcp_tools_cache: list | None = None

async def get_mcp_tools() -> list:
    global _mcp_tools_cache
    if _mcp_tools_cache is None:
        _mcp_tools_cache = await _mcp_client.get_tools()
    return _mcp_tools_cache

async def call_mcp_tool(tool_name: str, **kwargs):
    tools = await get_mcp_tools()
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        raise RuntimeError(f"MCP tool '{tool_name}' not found among loaded tools.")

    raw = await tool.ainvoke(kwargs)

    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        return json.loads(raw[0]["text"])
    return raw

class AgentState(TypedDict, total=False):
    question: str
    intake: IntakeClassification
    question_embedding: list[float]
    relevant_tables: list[TableSchema]
    golden_query_examples: list[GoldenQueryExample]
    resolved_entities: list[ResolvedEntity]
    metric_definitions: dict[str, str]
    context: RetrievedContext
    plan: QueryPlan
    sql: str
    validation: ValidationResult
    sql_retry_count: int
    approved: bool | None
    reviewer_note: str | None
    raw_rows: list[dict]
    report: BIInsightReport

INTAKE_CLASSIFIER_PROMPT = """You are classifying a business question before it becomes sql
Question: {question}
Classify it as one of:
- simple_lookup: a single fact, count, or filter
- multi_step_aggregation: requires grouping, joining, or ranking across tables
- out_of_scope: question is not answerable from the business database

Give a one-sentence reason for your classification.
"""

async def intake_classifier(state: AgentState) -> dict:
    llm = get_structured_llm(IntakeClassification, lite=True)
    result = await llm.ainvoke(INTAKE_CLASSIFIER_PROMPT.format(question=state["question"]))
    return {"intake": result}

async def embed_question(state: AgentState) -> dict:
    vector = await _embedder.aembed_query(state["question"])
    return {"question_embedding": vector}

async def schema_pruner(state: AgentState) -> dict:
    collection = _chroma_client.get_collection("schema_catalog")
    results = collection.query(query_embeddings=[state["question_embedding"]], n_results=SCHEMA_TOP_K)
    table_names = [metadata["table_name"] for metadata in results["metadatas"][0]]

    relevant_tables = []
    for table_name in table_names:
        raw = await call_mcp_tool("get_table_schema", table_name=table_name)
        if "error" in raw:
            continue

        relevant_tables.append(
            TableSchema(
                table_name=raw["table_name"],
                description=raw["description"],
                columns=[ColumnInfo(**col) for col in raw["columns"]],
            )
        )
    return {"relevant_tables": relevant_tables}

async def golden_query_retriever(state: AgentState) -> dict:
    collection = _chroma_client.get_collection("golden_queries")
    results = collection.query(query_embeddings=[state["question_embedding"]], n_results=GOLDEN_QUERY_TOP_K)

    examples = [
        GoldenQueryExample(question=document, sql=metadata["sql"], description=metadata["description"]) for document, metadata in zip(results["documents"][0], results["metadatas"][0])
    ]
    return {"golden_query_examples": examples}

async def entity_resolver(state: AgentState) -> dict:
    collection = _chroma_client.get_collection("entity_values")
    results = collection.query(query_embeddings=[state["question_embedding"]], n_results=ENTITY_TOP_K)

    resolved = [
        ResolvedEntity(
            mentioned_as=document,
            table_name=metadata["table_name"],
            column_name=metadata["column_name"],
            exact_value=metadata["exact_value"],
        )
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if distance <= ENTITY_MAX_DISTANCE
    ]
    return {"resolved_entities": resolved}

async def metric_definition_lookup(state: AgentState) -> dict:
    collection = _chroma_client.get_collection("business_glossary")
    results = collection.query(query_embeddings=[state["question_embedding"]], n_results=GLOSSARY_TOP_K)

    definitions = {
        document: metadata["definition"]
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if distance <= GLOSSARY_MAX_DISTANCE
    }
    return {"metric_definitions": definitions}

async def assemble_context(state: AgentState) -> dict:
    context = RetrievedContext(
        relevant_tables=state["relevant_tables"],
        golden_query_examples=state["golden_query_examples"],
        resolved_entities=state["resolved_entities"],
        metric_definitions=state["metric_definitions"],
    )
    return {"context": context}

def _format_tables(tables: list[TableSchema]) -> str:
    lines = []
    for table in tables:
        columns = ", ".join(f"{col.name} ({col.sql_type})" for col in table.columns)
        lines.append(f"- {table.table_name}: ({table.description})\n Columns: {columns}")
    return "\n".join(lines) if lines else "(none retrieved)"

def _format_golden_queries(examples: list[GoldenQueryExample]) -> str:
    lines = [f'- "{ex.question}" -> {ex.sql}' for ex in examples]
    return "\n".join(lines) if lines else "(none retrieved)"

def _format_entities(entities: list[ResolvedEntity]) -> str:
    lines = [f"- {e.table_name}.{e.column_name} = '{e.exact_value}'" for e in entities]
    return "\n".join(lines) if lines else "(none retrieved)"

def _format_metrics(metrics: dict[str, str]) -> str:
    lines = [f"- {term}: {definition}" for term, definition in metrics.items()]
    return "\n".join(lines) if lines else "(none retrieved)"

QUERY_PLANNER_PROMPT = """
You are planning an SQL query against a business database. Do not write SQL yet- describe the plan.
Question: {question}

Available tables: 
{tables}

Similar past queries for reference:
{golden_queries}

Resolved entity values (use these exact values, not the user's wording):
{entities}

Business metric definitions (use these exact definitions, not the user's wording):
{metrics}

Describe: Which tables are needed, what filters apply, what aggregations are needed, what joins connect the tables, and a one-sentence summary of intent.
"""

async def query_planner(state: AgentState) -> dict:
    llm = get_structured_llm(QueryPlan)
    context = state["context"]
    prompt = QUERY_PLANNER_PROMPT.format(
        question=state["question"],
        tables=_format_tables(context.relevant_tables),
        golden_queries=_format_golden_queries(context.golden_query_examples),
        entities=_format_entities(context.resolved_entities),
        metrics=_format_metrics(context.metric_definitions),
    )
    plan = await llm.ainvoke(prompt)
    return {"plan": plan}

class SqlOutput(BaseModel):
    sql: str = Field(description="A single SQLite SELECT statement.")

SQL_GENERATOR_PROMPT = """Write an SQLite SELECT query for this plan. Output only the SQL.

Question: {question}
Plan: {plan}

Available tables:
{tables}

Similar past queries for reference:
{golden_queries}

Resolved entity values (use these exact values):
{entities}

Business metric definitions:
{metrics}
{retry_feedback}
"""

RETRY_FEEDBACK_TEMPLATE = """
Your previous attempt was rejected:
SQL: {previous_sql}
Errors: {errors}
Fix these specific issues.
"""

async def sql_generator(state: AgentState) -> dict:
    llm = get_structured_llm(SqlOutput)
    context = state["context"]
    validation = state.get("validation")

    retry_feedback = ""
    if validation is not None and not validation.is_valid:
        retry_feedback = RETRY_FEEDBACK_TEMPLATE.format(
            previous_sql=state.get("sql", ""), errors="; ".join(validation.errors)
        )

    prompt = SQL_GENERATOR_PROMPT.format(
        question=state["question"],
        plan=state["plan"].intent_summary,
        tables=_format_tables(context.relevant_tables),
        golden_queries=_format_golden_queries(context.golden_query_examples),
        entities=_format_entities(context.resolved_entities),
        metrics=_format_metrics(context.metric_definitions),
        retry_feedback=retry_feedback,
    )
    result = await llm.ainvoke(prompt)

    retry_count = state.get("sql_retry_count", 0) + (1 if validation is not None else 0)
    return {"sql": result.sql, "sql_retry_count": retry_count}

RESTRICTED_COLUMNS = {
    ("customers", "email"),
    ("customers", "first_name"),
    ("customers", "last_name"),
}

FORBIDDEN_KEYWORDS = ("insert", "update", "delete", "drop", "alter", "attach", "pragma", "create")

def _connect_readonly() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{BUSINESS_DB_PATH}?mode=ro", uri=True)

def _references_restricted_column(sql: str) -> bool:
    lowered = sql.lower()
    return any(column.lower() in lowered for _, column in RESTRICTED_COLUMNS)

async def sql_validator(state: AgentState) -> dict:
    sql = state["sql"].strip().rstrip(";")
    errors = []

    stripped_lower = sql.lower().lstrip()
    if not stripped_lower.startswith("select"):    
        errors.append("Only SELECT statements are allowed.")
    if any(keyword in stripped_lower for keyword in FORBIDDEN_KEYWORDS):
        errors.append("Forbidden keyword detected.")

    if errors:
        return {"validation": ValidationResult(
            is_valid=False,
            errors=errors,
            touches_restricted_column=False,
            estimated_row_count=None,
            estimated_cost_score=1.0
        )}

    conn = _connect_readonly()
    row_count, db_errors = None, []
    try:
        conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        row_count = conn.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0]
    except sqlite3.Error as db_error:
        db_errors.append(str(db_error))
    finally:
        conn.close()

    cost_score = min((row_count or 0) / HIGH_ROW_COUNT_THRESHOLD, 1.0) if row_count is not None else 1.0
    return {"validation": ValidationResult(
        is_valid=len(db_errors) == 0,
        errors=db_errors,
        touches_restricted_column=_references_restricted_column(sql),
        estimated_row_count=row_count,
        estimated_cost_score=cost_score,
    )}

async def human_approval_interrupt(state: AgentState) -> dict:
    validation = state["validation"]
    reason = "touches a restricted column" if validation.touches_restricted_column else "estimated cost exceeds threshold"

    approval_request = ApprovalRequest(sql=state["sql"], reason=reason, validation=validation)
    decision_data = interrupt(approval_request.model_dump())
    decision = ApprovalDecision(**decision_data)

    return {"approved": decision.approved, "reviewer_note": decision.reviewer_note}

EXECUTOR_ROW_LIMIT = int(os.environ.get("EXECUTOR_ROW_LIMIT", "100"))

async def sql_executor(state: AgentState) -> dict:
    conn = _connect_readonly()
    try:
        cursor = conn.execute(state["sql"])
        columns = [description[0] for description in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchmany(EXECUTOR_ROW_LIMIT)]
    finally:
        conn.close()
    return {"raw_rows": rows}

class ChartRecommendation(BaseModel):
    should_chart: bool = Field(description="False for a single aggregate number or anything that doesn't compare multiple items.")
    mark_type: Literal["bar", "line", "point", "arc"] | None = None
    x_field: str | None = Field(default=None, description="Exact column name from the results to use as the x-axis / category.")
    y_field: str | None = Field(default=None, description="Exact column name from the results to use as the y-axis / value.")

class SynthesizedInsight(BaseModel):
    narrative_summary: str
    chart: ChartRecommendation
    confidence_note: str | None = None

RESULT_SYNTHESIZER_PROMPT = """Summarize these query results in plain English for a business stakeholder.

Question: {question}
SQL used: {sql}
Results ({row_count} rows, showing up to 20):
{rows_preview}

Give a one-to-two sentence narrative takeaway. Then decide whether this data suits a chart:
- Single number or nothing to compare: should_chart = false, explain why in confidence_note.
- Otherwise: should_chart = true, pick mark_type (bar, line, point, or arc), and name the exact result column for x_field and y_field."""

def _build_chart_spec(rows: list[dict], chart: ChartRecommendation) -> dict:
    if not chart.should_chart or not rows or not chart.x_field or not chart.y_field:
        return {}
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": chart.mark_type or "bar",
        "data": {"values": rows},
        "encoding": {
            "x": {"field": chart.x_field, "type": "nominal"},
            "y": {"field": chart.y_field, "type": "quantitative"},
        },
    }

async def result_synthesizer(state: AgentState) -> dict:
    llm = get_structured_llm(SynthesizedInsight)
    rows = state["raw_rows"]
    rows_preview = "\n".join(str(row) for row in rows[:20]) if rows else "(no rows returned)"

    prompt = RESULT_SYNTHESIZER_PROMPT.format(
        question=state["question"], sql=state["sql"], row_count=len(rows), rows_preview=rows_preview,
    )
    insight = await llm.ainvoke(prompt)

    report = BIInsightReport(
        narrative_summary=insight.narrative_summary,
        chart_spec=_build_chart_spec(rows[:20], insight.chart),
        raw_data_preview=rows[:20],
        sql_used=state["sql"],
        confidence_note=insight.confidence_note,
    )
    return {"report": report}

# NEW: Postgres Init Audit DB (Replaces the SQLite initialization)
async def init_audit_db() -> None:
    if not DATABASE_URL:
        return
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_audit_log (
                id SERIAL PRIMARY KEY,
                question TEXT,
                sql TEXT,
                is_valid BOOLEAN,
                touches_restricted_column BOOLEAN,
                estimated_row_count INTEGER,
                approved BOOLEAN,
                reviewer_note TEXT,
                row_count_returned INTEGER,
                logged_at TIMESTAMPTZ DEFAULT now()
            )
        """)

# NEW: Postgres Audit Logger
async def audit_logger(state: AgentState) -> dict:
    if not DATABASE_URL:
        return {}
    
    validation = state.get("validation")
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await conn.execute(
            "INSERT INTO query_audit_log "
            "(question, sql, is_valid, touches_restricted_column, estimated_row_count, "
            " approved, reviewer_note, row_count_returned) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                state["question"],
                state.get("sql", ""),  # The previous bug fix carried over!
                validation.is_valid if validation else None,
                validation.touches_restricted_column if validation else None,
                validation.estimated_row_count if validation else None,
                state.get("approved"),
                state.get("reviewer_note"),
                len(state.get("raw_rows", [])),
            ),
        )
    return {}

async def handle_out_of_scope(state: AgentState) -> dict:
    report = BIInsightReport(
        narrative_summary="This question can't be answered from the business database — it's out of scope for what's available.",
        chart_spec={}, raw_data_preview=[], sql_used="",
        confidence_note=state["intake"].reasoning,
    )
    return {"report": report}

async def handle_validation_failure(state: AgentState) -> dict:
    validation = state["validation"]
    report = BIInsightReport(
        narrative_summary="Couldn't generate a valid query for this question after multiple attempts.",
        chart_spec={}, raw_data_preview=[], sql_used=state["sql"],
        confidence_note="; ".join(validation.errors),
    )
    return {"report": report}

async def handle_approval_rejected(state: AgentState) -> dict:
    report = BIInsightReport(
        narrative_summary="This query was not approved for execution.",
        chart_spec={}, raw_data_preview=[], sql_used=state["sql"],
        confidence_note=state.get("reviewer_note"),
    )
    return {"report": report}

def route_after_intake(state: AgentState) -> str:
    if state["intake"].question_type == "out_of_scope":
        return "handle_out_of_scope"
    return "embed_question"

def route_after_validation(state: AgentState) -> str:
    validation = state["validation"]
    if not validation.is_valid:
        if state.get("sql_retry_count", 0) >= SQL_GEN_MAX_RETRIES:
            return "handle_validation_failure"
        return "sql_generator"
    if validation.touches_restricted_column or validation.estimated_cost_score >= 1.0:
        return "human_approval_interrupt"
    return "sql_executor"

def route_after_approval(state: AgentState) -> str:
    return "sql_executor" if state["approved"] else "handle_approval_rejected"

# NEW: Postgres Checkpointer Context
def get_checkpointer_context():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set.")
    return AsyncPostgresSaver.from_conn_string(DATABASE_URL)

def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    for name, fn in [
        ("intake_classifier", intake_classifier),
        ("handle_out_of_scope", handle_out_of_scope),
        ("embed_question", embed_question),
        ("schema_pruner", schema_pruner),
        ("golden_query_retriever", golden_query_retriever),
        ("entity_resolver", entity_resolver),
        ("metric_definition_lookup", metric_definition_lookup),
        ("assemble_context", assemble_context),
        ("query_planner", query_planner),
        ("sql_generator", sql_generator),
        ("sql_validator", sql_validator),
        ("human_approval_interrupt", human_approval_interrupt),
        ("handle_approval_rejected", handle_approval_rejected),
        ("handle_validation_failure", handle_validation_failure),
        ("sql_executor", sql_executor),
        ("result_synthesizer", result_synthesizer),
        ("audit_logger", audit_logger)
    ]: builder.add_node(name, fn)

    builder.add_edge(START, "intake_classifier")
    builder.add_conditional_edges("intake_classifier", route_after_intake)

    for target in ("schema_pruner", "golden_query_retriever", "entity_resolver", "metric_definition_lookup"):
        builder.add_edge("embed_question", target)
        builder.add_edge(target, "assemble_context")

    builder.add_edge("assemble_context", "query_planner")
    builder.add_edge("query_planner", "sql_generator")
    builder.add_edge("sql_generator", "sql_validator")
    builder.add_conditional_edges("sql_validator", route_after_validation)
    builder.add_conditional_edges("human_approval_interrupt", route_after_approval)

    builder.add_edge("sql_executor", "result_synthesizer")
    for terminal in ("result_synthesizer", "handle_out_of_scope", "handle_validation_failure", "handle_approval_rejected"):
        builder.add_edge(terminal, "audit_logger")
    builder.add_edge("audit_logger", END)

    return builder.compile(checkpointer=checkpointer)
