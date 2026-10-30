from types import SimpleNamespace

from app import retrieval


def test_vector_retrieval_uses_query_embedding_and_filters_below_threshold(monkeypatch):
    chunk = {"chunk_id": "chunk-1", "score": 0.72}
    monkeypatch.setattr(retrieval, "settings", SimpleNamespace(min_vector_similarity=0.5))
    monkeypatch.setattr(retrieval, "embed_query", lambda question: [0.1, 0.2])
    monkeypatch.setattr(retrieval.db, "vector_search", lambda vector, limit: [
        {"chunk_id": "chunk-1", "score": 0.72, "query_vector": vector, "limit": limit},
        {"chunk_id": "chunk-2", "score": 0.2},
    ])

    results = retrieval.retrieve("pergunta", "embedding", 3)

    assert [item["chunk_id"] for item in results] == [chunk["chunk_id"]]
    assert results[0]["query_vector"] == [0.1, 0.2]
    assert results[0]["limit"] == 3
