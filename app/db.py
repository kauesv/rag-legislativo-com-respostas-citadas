import re
from contextlib import closing
from functools import lru_cache

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from rank_bm25 import BM25Okapi

from app.settings import settings


class DatabaseUnavailable(RuntimeError):
    """Erro de conexão ou consulta ao banco tratado como indisponibilidade."""
    pass


def _tokens(text: str) -> list[str]:
    """Converte texto em tokens Unicode normalizados para a busca BM25."""
    return re.findall(r"[\wÀ-ÿ]+", text.casefold())


def connect(register=True):
    """Abre conexão PostgreSQL e, por padrão, registra vetores e linhas como dicts."""
    if not settings.database_url:
        raise DatabaseUnavailable("Configure DATABASE_URL e inicie o PostgreSQL com pgvector.")
    try:
        conn = psycopg.connect(settings.database_url, connect_timeout=5)
        if register:
            register_vector(conn)
            conn.commit()
            conn.row_factory = dict_row
        return conn
    except psycopg.Error as exc:
        if "conn" in locals():
            conn.close()
        raise DatabaseUnavailable(f"Banco indisponível: {exc.__class__.__name__}") from exc


def initialize_schema() -> None:
    """Cria a extensão pgvector, tabelas e índices usados pela aplicação."""
    with closing(connect(register=False)) as conn, conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                reference TEXT,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                publication_date DATE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS reference TEXT")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                embedding vector({settings.embedding_dimensions}) NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id)")


def save_document(conn, document: dict, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Atualiza documento e substitui seus chunks e embeddings na mesma transação."""
    if len(chunks) != len(embeddings):
        raise ValueError("Cada chunk precisa de um embedding correspondente.")
    with conn.transaction():
        conn.execute(
            """INSERT INTO documents(document_id, reference, title, url, publication_date) VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT(document_id) DO UPDATE SET reference=EXCLUDED.reference, title=EXCLUDED.title, url=EXCLUDED.url,
               publication_date=EXCLUDED.publication_date, updated_at=now()""",
            (document["document_id"], document["reference"], document["title"], document["url"], document.get("date")),
        )
        conn.execute("DELETE FROM chunks WHERE document_id=%s", (document["document_id"],))
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks(chunk_id, document_id, content, embedding) VALUES (%s,%s,%s,%s)",
                [(chunk["chunk_id"], document["document_id"], chunk["text"], Vector(vector)) for chunk, vector in zip(chunks, embeddings)],
            )


def vector_search(vector: list[float], limit: int) -> list[dict]:
    """Retorna os chunks mais próximos do vetor usando distância cosseno."""
    try:
        vector = Vector(vector)
        with closing(connect()) as conn:
            rows = conn.execute("""
                SELECT c.chunk_id, d.document_id, COALESCE(d.reference, d.document_id) AS reference,
                       d.title, d.url, d.publication_date::text AS date, c.content AS text,
                       1 - (c.embedding <=> %s) AS score
                FROM chunks c JOIN documents d USING (document_id)
                ORDER BY c.embedding <=> %s LIMIT %s
            """, (vector, vector, limit)).fetchall()
    except psycopg.Error as exc:
        raise DatabaseUnavailable("Falha ao consultar o índice vetorial; confira schema e dimensão dos embeddings.") from exc
    return [dict(row) for row in rows]


@lru_cache(maxsize=1)
def _bm25_index(chunk_count: int, latest_update):
    """Carrega chunks e monta índice BM25, armazenado em cache por versão do corpus."""
    with closing(connect()) as conn:
        rows = conn.execute("""
            SELECT c.chunk_id, d.document_id, COALESCE(d.reference, d.document_id) AS reference,
                   d.title, d.url, d.publication_date::text AS date, c.content AS text
            FROM chunks c JOIN documents d USING (document_id)
        """).fetchall()
    if not rows:
        return [], None
    index = BM25Okapi([_tokens(row["text"]) for row in rows])
    return rows, index


def bm25_search(query: str, limit: int) -> list[dict]:
    """Classifica chunks do corpus pela relevância lexical BM25."""
    try:
        with closing(connect()) as conn:
            state = conn.execute("""
                SELECT count(*) AS chunk_count, max(d.updated_at) AS latest_update
                FROM chunks c JOIN documents d USING (document_id)
            """).fetchone()
        if not state["chunk_count"]:
            return []
        rows, index = _bm25_index(state["chunk_count"], state["latest_update"])
    except psycopg.Error as exc:
        raise DatabaseUnavailable("Falha ao consultar o índice BM25; confira o schema do banco.") from exc
    if not rows:
        return []
    scores = index.get_scores(_tokens(query))
    ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)[:limit]
    return [dict(row) | {"score": float(score)} for row, score in ranked]
