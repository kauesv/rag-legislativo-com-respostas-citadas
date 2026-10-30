from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.settings import settings


class AskRequest(BaseModel):
    """Parâmetros validados para uma consulta à API."""
    question: str = Field(min_length=1, max_length=2000, description="Pergunta em português sobre proposições legislativas.")
    top_k: int = Field(default=5, ge=1, le=settings.max_top_k, description="Número máximo de trechos, limitado por MAX_TOP_K.")
    retrieval: Literal["bm25", "embedding"] = Field(default="embedding", description="Estratégia lexical BM25 ou busca vetorial.")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Remove espaços nas extremidades e rejeita perguntas em branco."""
        value = value.strip()
        if not value:
            raise ValueError("A pergunta não pode ficar vazia.")
        return value


class Source(BaseModel):
    """Referência oficial associada a um trecho citado na resposta."""
    document_id: str
    reference: str
    title: str
    url: str
    date: str | None = None
    chunk_id: str


class Timings(BaseModel):
    """Tempos de recuperação e geração retornados pela API, em milissegundos."""
    retrieval: float
    generation: float


class AskResponse(BaseModel):
    """Resposta da consulta, estado de abstenção, fontes e tempos."""
    answer: str
    abstained: bool
    sources: list[Source]
    timings_ms: Timings
