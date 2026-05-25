"""
MediExplain SG — Search Tools

Direct pgvector search functions used by the RAG pipeline.
Embeds a query and returns the most similar HealthHub article chunks.
"""

import json
import time

import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


def search_by_title(
    conn,
    entity: str,
    category: str | None = None,
) -> list[dict]:
    """
    Exact title match lookup — used for medication/condition names where
    semantic search returns wrong articles (e.g. 'Gliclazide' → 'Ezetimibe').
    Returns up to 1 result with similarity=1.0 if found, empty list otherwise.
    """
    if category:
        sql = """
            SELECT title, url, chunk_text FROM knowledge_base
            WHERE title ILIKE %s AND category = %s
            LIMIT 1
        """
        params = [f"%{entity}%", category]
    else:
        sql = """
            SELECT title, url, chunk_text FROM knowledge_base
            WHERE title ILIKE %s
            LIMIT 1
        """
        params = [f"%{entity}%"]

    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    if not row:
        return []

    title, url, chunk_text = row
    log.info("title_match_found", entity=entity, title=title)
    return [{"title": title, "url": url, "excerpt": chunk_text[:150], "similarity": 1.0}]


def search_direct(
    conn,
    embed_model: SentenceTransformer,
    query: str,
    category: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Embed the query, search pgvector for the most similar chunks,
    and return results as a list of dicts.
    """
    log.info("pgvector_search", query=query[:80], category=category, top_k=top_k)
    t_start = time.perf_counter()

    embedding = embed_model.encode(query, normalize_embeddings=True).tolist()

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
        log.info("pgvector_search_empty", query=query[:80], latency_ms=latency_ms)
        return []

    results = [
        {
            "title": title,
            "url": url,
            "excerpt": chunk_text[:150],
            "similarity": round(float(sim), 3),
        }
        for title, url, chunk_text, sim in rows
    ]

    log.info(
        "pgvector_search_done",
        query=query[:80],
        category=category,
        results=len(results),
        top_similarity=results[0]["similarity"],
        latency_ms=latency_ms,
    )
    return results
