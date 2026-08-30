import sqlite3
from mcp.server.fastmcp import FastMCP
from dataset import BUSINESS_DB_PATH, SCHEMA

mcp = FastMCP("business-db-catalog")

_TABLE_DESCRIPTIONS: dict[str, str] = {table.name: table.description for table in SCHEMA}
_COLUMN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    table.name: {col.name: col.description for col in table.columns} for table in SCHEMA
}


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{BUSINESS_DB_PATH}?mode=ro", uri=True)

def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone()
    return row is not None

def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(col[1] == column_name for col in columns)


@mcp.tool()
def list_tables() -> list[dict]:
    """Lists every table in the business database, with a one-line description of each."""
    conn = _connect()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    conn.close()
    return [{"table_name": name, "description": _TABLE_DESCRIPTIONS.get(name, "")} for (name,) in rows]


@mcp.tool()
def get_table_schema(table_name: str) -> dict:
    """Returns every column in a table — its name, SQL type, and description."""
    conn = _connect()
    if not _table_exists(conn, table_name):
        conn.close()
        return {"error": f"Table '{table_name}' does not exist."}

    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    conn.close()

    descriptions = _COLUMN_DESCRIPTIONS.get(table_name, {})
    return {
        "table_name": table_name,
        "description": _TABLE_DESCRIPTIONS.get(table_name, ""),
        "columns": [
            {"name": col[1], "sql_type": col[2], "description": descriptions.get(col[1], "")}
            for col in columns
        ],
    }

@mcp.tool()
def get_column_samples(table_name: str, column_name: str, limit: int = 10) -> list[str]:
    conn = _connect()
    if not _table_exists(conn, table_name) or not _column_exists(conn, table_name, column_name):
        conn.close()
        return []

    rows = conn.execute(f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [str(value) for (value,) in rows]

if __name__ == "__main__":
    mcp.run(transport="stdio")
