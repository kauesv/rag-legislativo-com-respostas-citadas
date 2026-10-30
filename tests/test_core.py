import pytest
from pydantic import ValidationError
from http.client import RemoteDisconnected
from types import SimpleNamespace

from app import chunking, ingest
from app.models import AskRequest


def test_ask_request_strips_question_and_applies_defaults():
    request = AskRequest(question="  Como tramita um projeto?  ")

    assert request.question == "Como tramita um projeto?"
    assert request.retrieval == "embedding"
    assert request.top_k == 5


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_request_rejects_empty_or_whitespace_question(question):
    with pytest.raises(ValidationError):
        AskRequest(question=question)


def test_ask_request_rejects_invalid_retrieval_strategy():
    with pytest.raises(ValidationError):
        AskRequest(question="Pergunta válida", retrieval="hybrid")


def test_chunk_text_normalizes_content_and_keeps_configured_overlap(monkeypatch):
    monkeypatch.setattr(chunking, "settings", SimpleNamespace(chunk_size=10, chunk_overlap=3))

    chunks = chunking.chunk_text("doc-1", "  abcde   fghij klmno  ")

    assert [chunk["text"] for chunk in chunks] == ["abcde fghi", "ghij klmno"]
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert chunks == chunking.chunk_text("doc-1", "abcde fghij klmno")


def test_chunk_text_rejects_overlap_equal_to_chunk_size(monkeypatch):
    monkeypatch.setattr(chunking, "settings", SimpleNamespace(chunk_size=10, chunk_overlap=10))

    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        chunking.chunk_text("doc-1", "some text")


def test_format_document_uses_official_metadata_without_downloading_pdf(monkeypatch):
    monkeypatch.setattr(ingest, "settings", SimpleNamespace(camara_base_url="https://dadosabertos.camara.leg.br/api/v2"))
    item = {
        "id": 123,
        "siglaTipo": "PL",
        "numero": 10,
        "ano": 2024,
        "ementa": "Cria uma política pública.",
        "uri": "https://dadosabertos.camara.leg.br/api/v2/proposicoes/123",
        "dataApresentacao": "2024-02-10T00:00:00",
        "statusProposicao": {"descricaoSituacao": "Em tramitação"},
    }

    document = ingest.format_document(item, include_pdfs=False)

    assert document["document_id"] == "123"
    assert document["reference"] == "PL 10/2024"
    assert document["date"] == "2024-02-10T00:00:00"
    assert "Cria uma política pública." in document["content"]
    assert "Em tramitação" in document["content"]


def test_format_document_skips_items_without_indexable_content():
    assert ingest.format_document({"id": 123}, include_pdfs=False) is None


def test_ingest_indexes_documents_and_closes_database_connection(monkeypatch):
    class Connection:
        closed = False

        def close(self):
            self.closed = True

    connection = Connection()
    pages = iter([
        {"dados": [{"id": 123, "siglaTipo": "PL", "numero": 10, "ano": 2024, "ementa": "Ementa."}]},
        {"dados": []},
    ])
    saved = []
    monkeypatch.setattr(ingest.db, "connect", lambda: connection)
    monkeypatch.setattr(ingest, "request_json", lambda _url: next(pages))
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(ingest.db, "save_document", lambda *_args: saved.append(True))

    result = ingest.ingest(1, "2024-01-01", "", [], include_pdfs=False)

    assert result == {"indexed": 1, "skipped": 0, "examined": 1}
    assert saved == [True]
    assert connection.closed


def test_ingest_uses_v2_details_and_skips_legacy_service(monkeypatch):
    class Connection:
        def close(self):
            pass

    item = {
        "id": 123,
        "siglaTipo": "PL",
        "numero": 10,
        "ano": 2024,
        "ementa": "Ementa da listagem.",
        "uri": "https://dadosabertos.camara.leg.br/api/v2/proposicoes/123",
    }
    pages = iter([
        {"dados": [item]},
        {"dados": {"urlInteiroTeor": "https://www.camara.leg.br/proposicoesWeb/integra.pdf"}},
    ])
    requested = []
    monkeypatch.setattr(ingest.db, "connect", lambda: Connection())
    monkeypatch.setattr(ingest, "request_json", lambda url: requested.append(url) or next(pages))
    monkeypatch.setattr(ingest, "extract_pdf_text", lambda _url: "Texto integral.")
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(ingest.db, "save_document", lambda *_args: None)

    ingest.ingest(1, "2024-01-01", "", [], include_pdfs=True)

    assert requested[-1] == item["uri"]
    assert len(requested) == 2


@pytest.mark.parametrize("limit", [0, 5001])
def test_ingest_rejects_limit_outside_supported_range(limit):
    with pytest.raises(ValueError, match="CAMARA_LIMITE"):
        ingest.ingest(limit, "2024-01-01", "", [], include_pdfs=False)


def test_request_bytes_retries_when_server_disconnects_before_responding(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def geturl(self):
            return "https://dadosabertos.camara.leg.br/api/v2/proposicoes"

        def read(self, _limit):
            return b"{}"

    attempts = 0

    def flaky_urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RemoteDisconnected("Remote end closed connection without response")
        return Response()

    monkeypatch.setattr(ingest, "urlopen", flaky_urlopen)
    monkeypatch.setattr(ingest.time, "sleep", lambda _delay: None)

    body = ingest.request_bytes(
        "https://dadosabertos.camara.leg.br/api/v2/proposicoes",
        "application/json",
        {"dadosabertos.camara.leg.br"},
    )

    assert body == b"{}"
    assert attempts == 2


def test_request_bytes_retries_gateway_timeout(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def geturl(self):
            return "https://dadosabertos.camara.leg.br/api/v2/proposicoes"

        def read(self, _limit):
            return b"{}"

    attempts = 0

    def gateway_timeout_then_success(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ingest.HTTPError("https://dadosabertos.camara.leg.br/api/v2/proposicoes", 504, "Gateway Timeout", {}, None)
        return Response()

    monkeypatch.setattr(ingest, "urlopen", gateway_timeout_then_success)
    monkeypatch.setattr(ingest.time, "sleep", lambda _delay: None)

    body = ingest.request_bytes(
        "https://dadosabertos.camara.leg.br/api/v2/proposicoes",
        "application/json",
        {"dadosabertos.camara.leg.br"},
    )

    assert body == b"{}"
    assert attempts == 3
