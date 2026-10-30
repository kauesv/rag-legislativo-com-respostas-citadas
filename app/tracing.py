import logging

from langfuse import get_client

from app.settings import settings

logger = logging.getLogger(__name__)


def _client():
    """Retorna o cliente Langfuse quando as duas credenciais estão configuradas."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    return get_client()


def flush_traces() -> None:
    try:
        client = _client()
        if client:
            client.flush()
    except Exception:
        logger.exception("Não foi possível enviar os traces pendentes ao Langfuse")


def start_embedding_observation():
    try:
        client = _client()
        return client.start_observation(
            as_type="embedding", name="document-embeddings", model=settings.embedding_model
        ) if client else None
    except Exception:
        logger.exception("Não foi possível iniciar o trace de embeddings no Langfuse")
        return None


def finish_embedding_observation(observation, usage: dict, text_count: int, success: bool = True, error_type: str | None = None) -> None:
    if observation is None:
        return
    try:
        observation.update(
            usage_details={"input": usage["input"]} if usage["input"] else None,
            metadata={"text_count": text_count, "success": success},
            **({} if success else {"level": "ERROR", "status_message": error_type or "Embedding failed"}),
        )
    except Exception:
        logger.exception("Não foi possível registrar o uso de embeddings no Langfuse")
    finally:
        try:
            observation.end()
        except Exception:
            logger.exception("Não foi possível finalizar o trace de embeddings no Langfuse")


def start_generation_observation():
    try:
        client = _client()
        return client.start_observation(
            as_type="generation", name="answer-generation", model=settings.chat_model
        ) if client else None
    except Exception:
        logger.exception("Não foi possível iniciar o trace de geração no Langfuse")
        return None


def finish_generation_observation(observation, usage: dict, success: bool, answer_chars: int = 0, citation_count: int = 0, error_type: str | None = None) -> None:
    if observation is None:
        return
    try:
        observation.update(
            usage_details=usage or None,
            metadata={"success": success, "answer_chars": answer_chars, "citation_count": citation_count},
            **({} if success else {"level": "ERROR", "status_message": error_type or "Generation failed"}),
        )
    except Exception:
        logger.exception("Não foi possível registrar o uso de geração no Langfuse")
    finally:
        try:
            observation.end()
        except Exception:
            logger.exception("Não foi possível finalizar o trace de geração no Langfuse")


def log_ask(strategy: str, top_k: int, chunks: list[dict], cited_ids: list[str], retrieval_ms: float, generation_ms: float, abstained: bool) -> None:
    """Registra metadados e tempos da consulta sem enviar o texto da pergunta."""
    try:
        client = _client()
        if client is None:
            return
        with client.start_as_current_observation(as_type="span", name="ask") as trace:
            trace.update(metadata={
                "retrieval": strategy,
                "model": settings.chat_model,
                "top_k": top_k,
                "retrieved_chunks": [{key: chunk.get(key) for key in ("chunk_id", "document_id", "reference", "score")} for chunk in chunks],
                "cited_chunk_ids": cited_ids,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "abstained": abstained,
            })
    except Exception:
        logger.exception("N\u00e3o foi poss\u00edvel registrar o trace no Langfuse")


def log_evaluation(strategy: str, metrics: dict, question_count: int, include_answers: bool) -> None:
    """Registra no Langfuse as métricas agregadas de uma avaliação golden set."""
    try:
        client = _client()
        if client is None:
            return
        with client.start_as_current_observation(
            as_type="span",
            name=f"golden-{strategy}",
            metadata={"retrieval": strategy, "golden_set": "v1", "questions": question_count, "answer_metrics": include_answers},
        ) as trace:
            trace.update(output={key: value for key, value in metrics.items() if value is not None})
        client.flush()
    except Exception:
        logger.exception("N\u00e3o foi poss\u00edvel registrar a avalia\u00e7\u00e3o no Langfuse")
