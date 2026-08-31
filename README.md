# Natural-Language BI Agent

**[Live Demo (Streamlit)](YOUR_STREAMLIT_URL_HERE)** | **[Backend API (Render)](https://natural-language-bi-agent.onrender.com)**

A multi-agent Text-to-SQL business intelligence pipeline that translates natural language into secure database operations, executes them against a read-only SQLite database, and synthesizes the output into narrative summaries and Vega-Lite charts. 

Built in preparation for the Conquerors Software Technologies internship.

## System Architecture

The pipeline uses a LangGraph state machine exposed via a FastAPI backend, with a Streamlit frontend for user interaction. 

1. **Intake Classification:** Evaluates the query as a simple lookup, multi-step aggregation, or out-of-scope to route the graph execution efficiently.
2. **Context Retrieval (RAG):** Embeds the question to fetch relevant table schemas via Model Context Protocol (MCP), resolves exact database entities, retrieves business metric definitions, and pulls historical "golden queries" from ChromaDB.
3. **Query Planning & Generation:** Formulates the SQL based strictly on the retrieved schema and business rules.
4. **Deterministic Validation:** Runs an EXPLAIN QUERY PLAN on a read-only SQLite connection to catch syntax errors and estimate compute cost before execution. Triggers a self-correction loop with error feedback if validation fails.
5. **Human-in-the-Loop Security:** Queries touching restricted columns (e.g., customer PII) or exceeding row-count thresholds trigger a LangGraph interrupt. Execution pauses and state is saved to Postgres until an administrator approves the payload.
6. **Execution & Synthesis:** Evaluates the tabular results and dynamically generates a JSON specification for Vega-Lite chart rendering alongside a plain-English executive summary.

## Engineering Decisions & Optimizations

* **Persistent MCP Sessions:** Replaced LangChain's default stateless tool-calling with a persistent MultiServerMCPClient session. This eliminates process-spawning overhead (re-importing dependencies on every tool call), preventing CPU starvation and 5-minute timeouts on Render's free tier.
* **Database Connection Pooling:** Implemented AsyncPostgresSaver with Supabase's Session Mode pooling (Port 5432 instead of the default 6543 Transaction pooler) to manage concurrent API requests without triggering psycopg3 prepared statement conflicts.
* **LLM Failover Runtime:** Custom runnable wrapper that attempts structured JSON extraction via Gemini 3.6 Flash, automatically degrading to a Groq open-source model via JSON mode if schema parsing fails.
* **Asynchronous Audit Logging:** Writes all query attempts, execution costs, and approval states to a remote Postgres audit table without blocking the main FastAPI event loop.

## Technology Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI, Python 3.12, Docker
* **Agent Framework:** LangGraph, LangChain
* **Models:** Gemini 3.6 Flash, Gemini 3.5 Flash-Lite, Groq
* **Vector Database:** ChromaDB (Local persistent)
* **Relational Databases:** SQLite (Read-only business data), PostgreSQL / Supabase (State checkpoints & Audit logs)
* **Tooling Protocol:** Model Context Protocol (MCP)

## Local Development Setup

### 1. Prerequisites
* Python 3.12+
* A Supabase Postgres project
* API Keys for Google Gemini and Groq

### 2. Environment Variables
Create a .env file in the root directory:

    DATABASE_URL=postgresql://postgres.[YOUR_PROJECT_REF]:[YOUR_PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
    GEMINI_API_KEY=your_gemini_key
    GROQ_API_KEY=your_groq_key
    SCHEMA_TOP_K=5
    SQL_GEN_MAX_RETRIES=3

*(Note: Ensure the Supabase connection string uses port 5432 for Session Mode to support psycopg3 prepared statements.)*

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

* **Backend (Render):** Deployed as a Web Service using the Docker runtime environment. Requires the .env variables mapped in the Render dashboard. 
* **Frontend (Streamlit Community Cloud):** Requires the Render backend URL added to the Streamlit Advanced Settings > Secrets block:

    API_BASE_URL = "https://natural-language-bi-agent.onrender.com"
