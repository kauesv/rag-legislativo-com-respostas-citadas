from app import db
from app.providers import embed_query
from app.settings import settings


def retrieve(question: str, strategy: str, top_k: int) -> list[dict]:
    """Recupera trechos por BM25 ou embeddings e aplica o limiar configurado."""
    if strategy == "bm25":
        rows = db.bm25_search(question, top_k)
        return [row for row in rows if row["score"] >= settings.min_bm25_score]
    if strategy == "embedding":
        vector = embed_query(question)
        rows = db.vector_search(vector, top_k)
        return [row for row in rows if row["score"] >= settings.min_vector_similarity]
    raise ValueError(f"Estratégia desconhecida: {strategy}")
