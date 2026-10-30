import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException

from app import db
from app.models import AskRequest, AskResponse, Source, Timings
from app.providers import ProviderError, generate_answer
from app.retrieval import retrieve
from app.settings import settings
from app.tracing import log_ask

ABSTENTION = "Não encontrei evidência suficiente nos documentos consultados."


@asynccontextmanager
async def lifespan(_app):
    """Tenta preparar o schema e mantém a API disponível para health se o banco cair."""
    try:
        db.initialize_schema()
    except Exception:
        logger.exception("Banco indisponível durante a inicialização; a API permanece acessível para health.")
    yield


app = FastAPI(
    title="RAG Legislativo com respostas citadas",
    description="Consulta proposições da Câmara com respostas limitadas às fontes recuperadas.",
    version="1.0.0",
    lifespan=lifespan,
)
logger = logging.getLogger(__name__)


@app.get("/health", tags=["operations"])
async def health():
    """Informa estado da API e presença de configuração de banco e modelo."""
    return {
        "status": "online",
        "database_configured": bool(settings.database_url),
        "model_configured": bool(settings.provider_api_key),
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["query"],
    summary="Responde com evidências de proposições oficiais",
    description="Busca trechos por BM25 ou embeddings. Quando não há evidência suficiente, retorna a frase de abstenção e sources vazio.",
    responses={502: {"description": "Falha do provedor de geração"}, 503: {"description": "Banco ou provedor de modelo indisponível"}},
)
def ask(request: AskRequest) -> AskResponse:
    """Recupera evidências, valida citações e monta a resposta HTTP da consulta."""
    retrieval_start = perf_counter()
    try:
        chunks = retrieve(request.question, request.retrieval, request.top_k)
    except db.DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    retrieval_ms = (perf_counter() - retrieval_start) * 1000

    if not chunks:
        response = AskResponse(
            answer=ABSTENTION,
            abstained=True,
            sources=[],
            timings_ms=Timings(retrieval=retrieval_ms, generation=0),
        )
        log_ask(request.retrieval, request.top_k, [], [], retrieval_ms, 0, True)
        return response

    generation_start = perf_counter()
    try:
        generated = generate_answer(request.question, chunks)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    generation_ms = (perf_counter() - generation_start) * 1000
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    citations = generated["citations"]
    if not generated["answer"] or not citations or any(chunk_id not in chunks_by_id for chunk_id in citations):
        response = AskResponse(
            answer=ABSTENTION,
            abstained=True,
            sources=[],
            timings_ms=Timings(retrieval=retrieval_ms, generation=generation_ms),
        )
        log_ask(request.retrieval, request.top_k, chunks, citations, retrieval_ms, generation_ms, True)
        return response

    selected = [chunks_by_id[chunk_id] for chunk_id in dict.fromkeys(citations)]
    sources = [Source(
        document_id=chunk["document_id"],
        reference=chunk.get("reference"),
        title=chunk["title"],
        url=chunk["url"],
        date=chunk.get("date"),
        chunk_id=chunk["chunk_id"],
    ) for chunk in selected]
    cited = "\n\nFontes:\n" + "\n".join(f"- [{source.reference or source.document_id}]({source.url})" for source in sources)
    response = AskResponse(
        answer=generated["answer"] + cited,
        abstained=False,
        sources=sources,
        timings_ms=Timings(retrieval=retrieval_ms, generation=generation_ms),
    )
    log_ask(request.retrieval, request.top_k, chunks, citations, retrieval_ms, generation_ms, False)
    return response
