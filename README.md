# Natural-Language BI Agent

**[Live Demo (Streamlit)](https://natural-language-bi-agent-drsmuzlysznzxqcs4nweis.streamlit.app)** | **[Backend API (Render)](https://natural-language-bi-agent.onrender.com)**

A multi-agent Text-to-SQL business intelligence pipeline that translates natural language into secure database operations, executes them against a read-only SQLite business database, tracks execution state and audit logs via PostgreSQL, and synthesizes the output into narrative summaries and Vega-Lite charts. 

Built in preparation for the Conquerors Software Technologies internship.

## System Architecture

The pipeline uses a LangGraph state machine exposed via a FastAPI backend, with a Streamlit frontend for user interaction. 

1. **Intake Classification:** Evaluates the query as a simple lookup, multi-step aggregation, or out-of-scope to route the graph execution efficiently.
2. **Context Retrieval (RAG):** Embeds the question to fetch relevant table schemas via Model Context Protocol (MCP), resolves exact database entities, retrieves business metric definitions, and pulls historical "golden queries" from ChromaDB.
3. **Query Planning & Generation:** Formulates the SQL based strictly on the retrieved schema and business rules.
4. **Deterministic Validation:** Runs an `EXPLAIN QUERY PLAN` on a read-only SQLite connection to catch syntax errors and estimate compute cost before execution. Triggers a self-correction loop with error feedback if validation fails.
5. **Human-in-the-Loop Security:** Queries touching restricted columns (e.g., customer PII) or exceeding row-count thresholds trigger a LangGraph interrupt. Execution pauses and state is saved to Postgres until a reviewer approves the query.
6. **Execution & Synthesis:** Evaluates the tabular results and dynamically generates a JSON specification for Vega-Lite chart rendering alongside a plain-English executive summary.

## Pipeline

The graph is deliberately not a straight line — it has a parallel fan-out, a bounded self-correction loop, and a conditional human-approval branch, not just sequential steps.

```mermaid
flowchart TD
    START([question]) --> intake[intake_classifier]
    intake -->|out_of_scope| oos[handle_out_of_scope]
    intake -->|else| embed[embed_question]

    subgraph Parallel Retrieval Fan-Out
        embed --> schema[schema_pruner]
        embed --> golden[golden_query_retriever]
        embed --> entity[entity_resolver]
        embed --> glossary[metric_definition_lookup]
    end

    schema --> assemble[assemble_context]
    golden --> assemble
    entity --> assemble
    glossary --> assemble

    assemble --> plan[query_planner]
    plan --> gen[sql_generator]
    gen --> validate[sql_validator]

    validate -->|invalid, retries left| gen
    validate -->|invalid, retries exhausted| fail[handle_validation_failure]
    validate -->|valid, restricted column or high cost| interrupt[human_approval_interrupt]
    validate -->|valid, safe| exec[sql_executor]

    interrupt -->|approved| exec
    interrupt -->|rejected| rejected[handle_approval_rejected]

    exec --> synth[result_synthesizer]

    synth --> audit[audit_logger]
    oos --> audit
    fail --> audit
    rejected --> audit
    audit --> END([response])
