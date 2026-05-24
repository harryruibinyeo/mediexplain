"""
MediExplain SG — LangChain Search Tools

Defines the two tools the LangChain agent uses to search the knowledge base.
Both tools query pgvector using cosine similarity on the embedded query text.

Tools are created via a factory function (make_tools) so they close over
the shared database connection and embedding model loaded at startup.
"""

import json
import time

import structlog
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


def make_tools(conn, embed_model: SentenceTransformer):
    """
    Factory that returns the two search tools bound to the live db connection
    and embedding model. Called once at API startup.
    """

    def _search(query: str, category: str | None, top_k: int = 5) -> str:
        """
        Embed the query, search pgvector for the most similar chunks,
        and return the results as a JSON string so the agent can read them.
        """
        log.info("pgvector_search", query=query, category=category, top_k=top_k)
        t_start = time.perf_counter()

        embedding = embed_model.encode(
            query, normalize_embeddings=True
        ).tolist()

        if category:
            sql = """
                SELECT title, url, chunk_text,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_base
                WHERE category = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [embedding, category, embedding, top_k]
        else:
            sql = """
                SELECT title, url, chunk_text,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_base
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [embedding, embedding, top_k]

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        latency_ms = round((time.perf_counter() - t_start) * 1000)

        if not rows:
            log.info("pgvector_search_empty", query=query, latency_ms=latency_ms)
            return "No relevant articles found."

        results = [
            {
                "title": title,
                "url": url,
                "excerpt": chunk_text[:400],
                "similarity": round(float(sim), 3),
            }
            for title, url, chunk_text, sim in rows
        ]

        log.info(
            "pgvector_search_done",
            query=query,
            results=len(results),
            top_similarity=results[0]["similarity"],
            latency_ms=latency_ms,
        )
        return json.dumps(results, ensure_ascii=False)

    @tool
    def search_conditions(query: str) -> str:
        """
        Search HealthHub articles about health conditions, diagnoses,
        symptoms, and test results. Use this when the document mentions
        a diagnosis, a lab value, or a medical condition.

        Example queries: 'HbA1c diabetes', 'kidney function creatinine',
        'high blood pressure hypertension'
        """
        return _search(query, category="health-condition")

    @tool
    def search_medications(query: str) -> str:
        """
        Search HealthHub articles about medications, dosages, and treatments.
        Use this when the document mentions a drug name or prescription.

        Example queries: 'metformin diabetes', 'atorvastatin cholesterol',
        'amlodipine blood pressure'
        """
        return _search(query, category="medication-devices-treatment")

    return [search_conditions, search_medications]
