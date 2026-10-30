from fastapi.testclient import TestClient

from app import main
from app.providers import ProviderError


CHUNK = {
    "chunk_id": "chunk-1",
    "document_id": "123",
    "reference": "PL 10/2024",
    "title": "Projeto sobre educação",
    "url": "https://dadosabertos.camara.leg.br/api/v2/proposicoes/123",
    "date": "2024-02-10",
    "text": "Trecho oficial.",
}


def client(monkeypatch):
    monkeypatch.setattr(main.db, "initialize_schema", lambda: None)
    return TestClient(main.app)


def test_ask_returns_answer_and_sources_for_valid_citation(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda *_args: [CHUNK])
    monkeypatch.setattr(main, "generate_answer", lambda *_args: {"answer": "Resposta sustentada.", "citations": ["chunk-1"]})
    monkeypatch.setattr(main, "log_ask", lambda *_args: None)

    with client(monkeypatch) as test_client:
        response = test_client.post("/ask", json={"question": "Qual é a proposta?"})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert body["sources"][0]["reference"] == "PL 10/2024"
    assert "Fontes:" in body["answer"]


def test_ask_abstains_when_retrieval_has_no_evidence(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda *_args: [])
    monkeypatch.setattr(main, "log_ask", lambda *_args: None)

    with client(monkeypatch) as test_client:
        response = test_client.post("/ask", json={"question": "Pergunta sem evidência"})

    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["sources"] == []
    assert response.json()["timings_ms"]["generation"] == 0


def test_ask_abstains_when_model_cites_unknown_chunk(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda *_args: [CHUNK])
    monkeypatch.setattr(main, "generate_answer", lambda *_args: {"answer": "Resposta sem referência válida.", "citations": ["unknown"]})
    monkeypatch.setattr(main, "log_ask", lambda *_args: None)

    with client(monkeypatch) as test_client:
        response = test_client.post("/ask", json={"question": "Qual é a proposta?"})

    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["sources"] == []


def test_ask_returns_503_when_retrieval_provider_is_unavailable(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda *_args: (_ for _ in ()).throw(ProviderError("indisponível")))

    with client(monkeypatch) as test_client:
        response = test_client.post("/ask", json={"question": "Qual é a proposta?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "indisponível"


def test_ask_rejects_invalid_request_with_422(monkeypatch):
    with client(monkeypatch) as test_client:
        response = test_client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
