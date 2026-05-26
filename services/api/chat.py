"""
MediExplain SG — NL2SQL Chat Pipeline

4-step pipeline:
  1. Generator  — Qwen reads the user question + table schema → writes a SELECT query
  2. Reviewer   — Python validates the SQL: must be SELECT-only, no destructive keywords
  3. Executor   — psycopg2 runs the query against the knowledge_base table
  4. Explainer  — Qwen reads the question + result rows → plain-English answer

No LangChain. Pure Python + httpx.
"""

import re
import time
import uuid

import httpx
import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Table schema shown to the Generator.
# The embedding column is deliberately hidden — it is a 768-dim vector that
# Qwen cannot use and would confuse the query generator.
# ---------------------------------------------------------------------------
TABLE_SCHEMA = """
Table: knowledge_base
Columns:
  id          INTEGER  — unique row ID (auto-increment)
  slug        TEXT     — URL-friendly article identifier (e.g. "amlodipine")
  url         TEXT     — full HealthHub article URL
  category    TEXT     — "health-condition" OR "medication-devices-treatment"
  title       TEXT     — article title (e.g. "Amlodipine", "Hypertension")
  chunk_index INTEGER  — chunk number within the article (0, 1, 2, ...)
  chunk_text  TEXT     — the actual article text content for this chunk

IMPORTANT:
- Each article is split into multiple chunks. The same title/slug/url appears
  in several rows with different chunk_index values.
- When listing or counting articles, use DISTINCT ON (title) or GROUP BY title
  to avoid counting the same article multiple times.
- Always add LIMIT 20 unless the question asks for a count.
- Only write SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, or ALTER.
- Do not reference the embedding column — it does not exist for your purposes.
"""

# ---------------------------------------------------------------------------
# Keywords that must never appear in executable SQL (case-insensitive).
# The Reviewer checks for these before any query touches the database.
# ---------------------------------------------------------------------------
_BLOCKED = {
    "drop", "delete", "update", "insert", "alter",
    "create", "truncate", "grant", "revoke", "execute",
    "pg_", "information_schema",
}


# ---------------------------------------------------------------------------
# Step 1 — Generator
# ---------------------------------------------------------------------------

async def _generate_sql(question: str, vllm_url: str, model_name: str) -> str:
    """
    Ask Qwen to write a PostgreSQL SELECT query for the user's question.
    Returns the raw SQL string (may include markdown fences — stripped below).
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{vllm_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a PostgreSQL query generator. "
                            "Given a table schema and a user question, write a single "
                            "PostgreSQL SELECT query that answers the question. "
                            "Return ONLY the SQL query — no explanation, no markdown, "
                            "no code fences. Just the raw SQL."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Table schema:\n{TABLE_SCHEMA}\n\n"
                            f"Question: {question}\n\n"
                            "Write the SQL query:"
                        ),
                    },
                ],
                "max_tokens": 200,
                "temperature": 0,
            },
            timeout=120.0,
        )
        response.raise_for_status()

    sql = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if Qwen wraps the output
    if sql.startswith("```"):
        lines = sql.split("\n")
        # Remove first line (```sql or ```) and last line (```)
        sql = "\n".join(lines[1:-1]).strip()

    return sql


# ---------------------------------------------------------------------------
# Step 2 — Reviewer
# ---------------------------------------------------------------------------

def _review_sql(sql: str) -> tuple[bool, str]:
    """
    Validate that the SQL is safe to execute.
    Returns (is_safe, reason).

    Rules:
    - Must start with SELECT (after stripping whitespace/comments)
    - Must not contain any blocked keywords (DROP, DELETE, etc.)
    - Must not contain multiple statements (semicolons mid-query)
    """
    normalised = sql.strip().lower()

    # Must start with SELECT
    if not normalised.startswith("select"):
        return False, f"Query does not start with SELECT: {sql[:80]}"

    # Check for blocked keywords — word boundary match to avoid false positives
    # (e.g. "created_at" should not match "create")
    for keyword in _BLOCKED:
        pattern = rf"\b{re.escape(keyword)}\b"
        if re.search(pattern, normalised):
            return False, f"Blocked keyword detected: {keyword}"

    # No multiple statements — semicolon only allowed at the very end
    stripped = normalised.rstrip("; \n\t")
    if ";" in stripped:
        return False, "Multiple SQL statements are not allowed"

    return True, "ok"


# ---------------------------------------------------------------------------
# Step 3 — Executor
# ---------------------------------------------------------------------------

def _execute_sql(conn, sql: str) -> list[dict]:
    """
    Run the SQL against the knowledge_base table.
    Returns up to 20 rows as a list of dicts (column_name → value).
    Rolls back on error and re-raises so the caller can handle it.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(20)
        return [dict(zip(columns, row)) for row in rows]
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Step 4 — Explainer
# ---------------------------------------------------------------------------

async def _explain_results(
    question: str,
    sql: str,
    rows: list[dict],
    vllm_url: str,
    model_name: str,
) -> str:
    """
    Ask Qwen to turn the SQL result rows into a plain-English answer.
    """
    if not rows:
        rows_text = "The query returned no results."
    else:
        # Format rows as a simple text table for the LLM
        rows_text = "\n".join(str(row) for row in rows)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{vllm_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant for Singapore patients. "
                            "A user asked a question about a medical knowledge base. "
                            "You have the query results. Answer the user's question "
                            "in plain, friendly English using the results. "
                            "If there are no results, say so clearly. "
                            "Be concise — 2 to 5 sentences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"SQL query used:\n{sql}\n\n"
                            f"Query results:\n{rows_text}\n\n"
                            "Please answer the question based on these results."
                        ),
                    },
                ],
                "max_tokens": 300,
                "temperature": 0,
            },
            timeout=120.0,
        )
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run(
    question: str,
    conn,
    vllm_url: str,
    model_name: str,
) -> dict:
    """
    Run the full NL2SQL pipeline and return the result.

    Returns:
      {
        "answer":   str,           # plain-English answer from Explainer
        "sql":      str,           # the SQL query the Generator wrote
        "rows":     list[dict],    # raw result rows from Executor
        "error":    str | None,    # set if Reviewer blocked the query
        "meta":     dict,          # request_id, latency_s, row_count
      }
    """
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()
    log.info("chat_start", request_id=request_id, question=question[:80])

    # ------------------------------------------------------------------
    # Step 1: Generate SQL
    # ------------------------------------------------------------------
    try:
        sql = await _generate_sql(question, vllm_url, model_name)
    except Exception as e:
        log.error("sql_generation_failed", request_id=request_id, error=str(e))
        return {
            "answer": "Sorry, I couldn't generate a query for that question. Please try rephrasing.",
            "sql": "",
            "rows": [],
            "error": str(e),
            "meta": {"request_id": request_id, "latency_s": 0, "row_count": 0},
        }

    log.info("sql_generated", request_id=request_id, sql=sql[:200])

    # ------------------------------------------------------------------
    # Step 2: Review SQL
    # ------------------------------------------------------------------
    is_safe, reason = _review_sql(sql)
    if not is_safe:
        log.warning("sql_blocked", request_id=request_id, reason=reason, sql=sql[:200])
        return {
            "answer": "I can only answer read-only questions about the medical knowledge base. Please ask something like 'What medications are available for diabetes?'",
            "sql": sql,
            "rows": [],
            "error": f"Blocked: {reason}",
            "meta": {"request_id": request_id, "latency_s": 0, "row_count": 0},
        }

    # ------------------------------------------------------------------
    # Step 3: Execute SQL
    # ------------------------------------------------------------------
    try:
        rows = _execute_sql(conn, sql)
    except Exception as e:
        log.error("sql_execution_failed", request_id=request_id, sql=sql[:200], error=str(e))
        return {
            "answer": f"The query ran into a database error. The generated SQL may be invalid. Error: {str(e)[:100]}",
            "sql": sql,
            "rows": [],
            "error": str(e),
            "meta": {"request_id": request_id, "latency_s": 0, "row_count": 0},
        }

    log.info("sql_executed", request_id=request_id, row_count=len(rows))

    # ------------------------------------------------------------------
    # Step 4: Explain results
    # ------------------------------------------------------------------
    try:
        answer = await _explain_results(question, sql, rows, vllm_url, model_name)
    except Exception as e:
        log.error("explanation_failed", request_id=request_id, error=str(e))
        answer = f"Query returned {len(rows)} result(s) but I couldn't generate an explanation."

    latency = round(time.perf_counter() - t_start, 2)
    log.info("chat_done", request_id=request_id, row_count=len(rows), latency_s=latency)

    return {
        "answer": answer,
        "sql":    sql,
        "rows":   rows,
        "error":  None,
        "meta": {
            "request_id": request_id,
            "latency_s":  latency,
            "row_count":  len(rows),
        },
    }
