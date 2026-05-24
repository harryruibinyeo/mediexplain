"""
MediExplain SG — Ingest Service

Reads the raw HealthHub JSON articles scraped from healthhub.sg,
splits each article body into overlapping text chunks, generates
embeddings for each chunk using sentence-transformers, and loads
everything into pgvector.

Run this once before starting the API service. Re-running is safe —
articles already in the database are skipped.

Environment variables:
  DATABASE_URL  — PostgreSQL connection string
  RAW_DATA_DIR  — path to scraped JSON files (default: /data/raw)
  EMBED_MODEL   — sentence-transformers model (default: all-mpnet-base-v2)
  CHUNK_SIZE    — max characters per chunk (default: 600)
  CHUNK_OVERLAP — overlap between chunks in characters (default: 100)
"""

import json
import os
import time
from pathlib import Path

import psycopg2
import structlog
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Structured logging — emits one JSON line per event, same pattern as scraper.
# Pipe output through `jq` to filter: e.g. `python main.py | jq 'select(.level=="error")'`
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config — all tunable via environment variables
# ---------------------------------------------------------------------------
DATABASE_URL  = os.environ["DATABASE_URL"]
RAW_DATA_DIR  = Path(os.environ.get("RAW_DATA_DIR", "/data/raw"))
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "all-mpnet-base-v2")
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# all-mpnet-base-v2 produces 768-dimensional vectors.
# This must match the vector(768) column in the database schema.
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Split article body into overlapping chunks.

    Strategy: split on double-newlines (paragraph breaks) first so chunks
    respect the natural sections of the article. If a paragraph itself
    exceeds `size`, fall back to hard character splitting.

    Overlapping chunks ensure that sentences near a boundary appear in
    two consecutive chunks, so the retrieval step never misses context
    that straddles a split point.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                # Carry the tail of the previous chunk forward for overlap
                current = current[-overlap:] if len(current) > overlap else current

            if len(para) > size:
                # Hard-split oversized paragraphs
                start = 0
                while start < len(para):
                    chunks.append(para[start : start + size])
                    start += size - overlap
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_schema(conn) -> None:
    """
    Create the knowledge_base table and IVFFlat vector index if they
    do not already exist.

    IVFFlat is an approximate nearest-neighbour index — much faster than
    an exact scan for large tables. `lists=100` is a sensible default for
    up to ~1 million rows.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id           SERIAL PRIMARY KEY,
                slug         TEXT NOT NULL,
                url          TEXT NOT NULL,
                category     TEXT NOT NULL,
                title        TEXT NOT NULL,
                chunk_index  INTEGER NOT NULL,
                chunk_text   TEXT NOT NULL,
                embedding    vector({EMBEDDING_DIM})
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx
            ON knowledge_base USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        conn.commit()
    log.info("schema_ready", table="knowledge_base", embedding_dim=EMBEDDING_DIM)


def already_ingested(conn, slug: str) -> bool:
    """Return True if this article slug is already present in the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM knowledge_base WHERE slug = %s LIMIT 1", (slug,))
        return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Load raw articles from disk
# ---------------------------------------------------------------------------

def load_articles(raw_dir: Path) -> list[dict]:
    """Read every JSON file in raw_dir and return a list of article dicts."""
    articles = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                articles.append(json.load(f))
        except Exception as e:
            log.warning("load_failed", path=str(path), error=str(e))

    log.info(
        "articles_loaded",
        count=len(articles),
        health_conditions=sum(1 for a in articles if a.get("category") == "health-condition"),
        medications=sum(1 for a in articles if a.get("category") == "medication-devices-treatment"),
    )
    return articles


# ---------------------------------------------------------------------------
# Ingest pipeline
# ---------------------------------------------------------------------------

def ingest(conn, model: SentenceTransformer, articles: list[dict]) -> None:
    """
    For each article:
      1. Skip if already in the database
      2. Chunk the body text
      3. Embed all chunks in one batch (GPU-accelerated if available)
      4. Insert every chunk + its vector into knowledge_base
      5. Log timing and chunk count
    """
    total_chunks = 0
    skipped = 0
    failed = 0

    for i, article in enumerate(articles, start=1):
        slug = article.get("slug", "unknown")

        if already_ingested(conn, slug):
            log.info("skipped_existing", slug=slug, current=i, total=len(articles))
            skipped += 1
            continue

        log.info("ingesting", slug=slug, category=article.get("category"), current=i, total=len(articles))
        t_start = time.perf_counter()

        try:
            chunks = chunk_text(article["body"], CHUNK_SIZE, CHUNK_OVERLAP)

            # Encode the whole batch at once — sentence-transformers uses
            # GPU automatically if torch.cuda.is_available()
            embeddings = model.encode(
                chunks,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,  # normalised vectors work best with cosine similarity
            )

            with conn.cursor() as cur:
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    cur.execute(
                        """
                        INSERT INTO knowledge_base
                            (slug, url, category, title, chunk_index, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            article["slug"],
                            article["url"],
                            article["category"],
                            article["title"],
                            idx,
                            chunk,
                            embedding.tolist(),
                        ),
                    )
            conn.commit()

            latency_ms = round((time.perf_counter() - t_start) * 1000)
            total_chunks += len(chunks)
            log.info("article_ingested", slug=slug, chunks=len(chunks), latency_ms=latency_ms)

        except Exception as e:
            conn.rollback()
            log.error("ingest_failed", slug=slug, error=str(e))
            failed += 1

    log.info(
        "ingest_complete",
        total_articles=len(articles),
        ingested=len(articles) - skipped - failed,
        skipped=skipped,
        failed=failed,
        total_chunks=total_chunks,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info(
        "ingest_starting",
        model=EMBED_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        raw_data_dir=str(RAW_DATA_DIR),
    )

    # Load the embedding model
    # sentence-transformers automatically uses your GPU if CUDA is available
    log.info("loading_embed_model", model=EMBED_MODEL)
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL)
    log.info(
        "embed_model_loaded",
        model=EMBED_MODEL,
        embedding_dim=model.get_sentence_embedding_dimension(),
        load_time_s=round(time.perf_counter() - t0, 2),
    )

    # Connect to PostgreSQL — hide credentials from logs
    db_display = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    log.info("connecting_db", host=db_display)
    conn = psycopg2.connect(DATABASE_URL)

    # Install the pgvector extension first, then register the vector type.
    # register_vector() fails if the extension isn't installed yet.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    register_vector(conn)
    log.info("db_connected")

    create_schema(conn)
    articles = load_articles(RAW_DATA_DIR)
    ingest(conn, model, articles)

    conn.close()
    log.info("ingest_done")


if __name__ == "__main__":
    main()
