"""
schemas.py

Pydantic models shared across the whole project — anything that crosses a
node boundary in graph.py, gets returned by an MCP tool in mcp_server.py, or
gets serialized in a FastAPI response in main.py lives here, once, so none
of those files need their own copy.

Does NOT include the LangGraph graph state itself — that's defined in
graph.py, next to the nodes that actually read and write it, since it's
wiring logic more than a reusable data model.
"""

from typing import Literal

from pydantic import BaseModel, Field


# --- intake_classifier output ---

class IntakeClassification(BaseModel):
    question_type: Literal["simple_lookup", "multi_step_aggregation", "out_of_scope"]
    reasoning: str = Field(description="One sentence on why this question falls into that category.")


# --- context_assembler output (schema_pruner + golden_query_retriever + entity_resolver + metric_definition_lookup, fanned in) ---

class ColumnInfo(BaseModel):
    name: str
    sql_type: str
    description: str


class TableSchema(BaseModel):
    table_name: str
    description: str
    columns: list[ColumnInfo]


class GoldenQueryExample(BaseModel):
    question: str
    sql: str
    description: str


class ResolvedEntity(BaseModel):
    mentioned_as: str = Field(description="The phrase the user actually used, e.g. 'California'")
    table_name: str
    column_name: str
    exact_value: str = Field(description="The exact value stored in the database, e.g. 'CA'")


class RetrievedContext(BaseModel):
    relevant_tables: list[TableSchema]
    golden_query_examples: list[GoldenQueryExample]
    resolved_entities: list[ResolvedEntity]
    metric_definitions: dict[str, str] = Field(
        description="Metric name (e.g. 'Churned Customer') mapped to its exact business definition."
    )


# --- query_planner output ---

class QueryPlan(BaseModel):
    target_tables: list[str]
    filters: list[str]
    aggregations: list[str]
    joins: list[str]
    intent_summary: str = Field(description="One sentence restating what the user is actually asking for.")


# --- sql_validator output (deterministic function, not an LLM call — still typed for consistency) ---

class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    touches_restricted_column: bool
    estimated_row_count: int | None = None
    estimated_cost_score: float = Field(
        ge=0.0, le=1.0, description="Normalized 0-1 heuristic — see sql_validator for how this is computed."
    )


# --- human_approval_interrupt payload / FastAPI resume endpoint ---

class ApprovalRequest(BaseModel):
    sql: str
    reason: str = Field(description="Why this query needs approval, e.g. 'touches restricted column: email'.")
    validation: ValidationResult


class ApprovalDecision(BaseModel):
    approved: bool
    reviewer_note: str | None = None


# --- result_synthesizer output ---

class BIInsightReport(BaseModel):
    narrative_summary: str
    chart_spec: dict = Field(description="Vega-Lite chart specification as a JSON-serializable dict.")
    raw_data_preview: list[dict]
    sql_used: str
    confidence_note: str | None = None
