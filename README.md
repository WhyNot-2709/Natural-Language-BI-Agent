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
```

* **Fan-out / fan-in:** The four retrieval nodes run in parallel off a single shared question embedding, not four separate embedding calls. `assemble_context` is the only node with in-edges from all four.
* **Retry loop:** `sql_validator` is deterministic, not another LLM call. It re-runs `sql_generator` with the specific error as feedback, capped at `SQL_GEN_MAX_RETRIES`.
* **HITL branch:** Only triggers on a restricted column or a high estimated row count. Everything else skips straight to execution.

## Engineering Decisions & Optimizations

* **Persistent MCP Sessions:** Replaced LangChain's default stateless tool-calling with a persistent `MultiServerMCPClient` session. This eliminates process-spawning overhead (re-importing dependencies on every tool call), preventing CPU starvation and 5-minute timeouts on Render's free tier.
* **Postgres Connection Mode:** Moved away from Supabase's transaction-mode pooler (Port `6543`) to session mode (Port `5432`) specifically to stop prepared-statement crashes (`psycopg3` conflict). You trade connection pooling for absolute stability—fine at this scale, though it would need revisiting under high concurrent production load.
* **LLM Failover Runtime:** Custom runnable wrapper that attempts structured JSON extraction via Gemini 3.6 Flash, automatically degrading to a Groq open-source model via JSON mode if schema parsing fails.
* **Asynchronous Audit Logging:** Writes all query attempts, estimated row counts, and approval states to a remote Postgres audit table without blocking the main FastAPI event loop.

## Technology Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI, Python 3.12, Docker
* **Agent Framework:** LangGraph, LangChain
* **Models:** Gemini 3.6 Flash, Gemini 3.5 Flash-Lite, Groq
* **Vector Database:** ChromaDB (Local persistent)
* **Relational Databases:** SQLite (Read-only business data), PostgreSQL / Supabase (State checkpoints & Audit logs)
* **Tooling Protocol:** Model Context Protocol (MCP)

## Known Limitations

* `promotions` has no foreign key to `orders` — promo-effectiveness questions aren't answerable with the current schema.
* `customers.is_active` is a snapshot, not a churn-event timestamp — trend questions ("is churn getting worse?") can't be answered correctly from this data, only point-in-time splits.
* Restricted-column detection in `sql_validator` is a substring check on the generated SQL, not a real SQL parser — sufficient for this schema, not adversarially robust.
* `human_approval_interrupt` has no auth — any caller can approve or reject a paused query. Fine for a demo, not for production.

## Local Development Setup

### 1. Prerequisites
* Python 3.12+
* A Supabase Postgres project
* API Keys for Google Gemini and Groq

### 2. Environment Variables
Create a `.env` file in the root directory:

    DATABASE_URL=postgresql://postgres.[YOUR_PROJECT_REF]:[YOUR_PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
    GEMINI_API_KEY=your_gemini_key
    GROQ_API_KEY=your_groq_key
    SCHEMA_TOP_K=5
    SQL_GEN_MAX_RETRIES=3

*(Note: Ensure the Supabase connection string uses port `5432` for Session Mode to support `psycopg3` prepared statements.)*

### 3. Installation

    git clone https://github.com/WhyNot-2709/Natural-Language-BI-Agent.git
    cd Natural-Language-BI-Agent
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

### 4. Running the Application
Start the FastAPI backend (handles graph execution and MCP servers):

    uvicorn main:app --reload

In a new terminal pane, start the Streamlit frontend:

    streamlit run streamlit_app.py

## Deployment

* **Backend (Render):** Deployed as a Web Service using the Docker runtime environment. Requires the `.env` variables mapped in the Render dashboard. 
* **Frontend (Streamlit Community Cloud):** Requires the Render backend URL added to the Streamlit Advanced Settings > Secrets block:

    API_BASE_URL = "https://natural-language-bi-agent.onrender.com"
