import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Configuração tipada da aplicação, carregada das variáveis de ambiente."""

    database_url: str = os.getenv("DATABASE_URL", "")
    provider_base_url: str = os.getenv("MODEL_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    provider_api_key: str = os.getenv("MODEL_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-6-luna")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    min_vector_similarity: float = float(os.getenv("MIN_VECTOR_SIMILARITY", "0.35"))
    min_bm25_score: float = float(os.getenv("MIN_BM25_SCORE", "0.5"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "10"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "180"))
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    camara_base_url: str = os.getenv("CAMARA_API_BASE_URL", "https://dadosabertos.camara.leg.br/api/v2").rstrip("/")
    ingest_start: str = os.getenv("CAMARA_DATA_INICIO", "2020-01-01")
    ingest_end: str = os.getenv("CAMARA_DATA_FIM", "")
    ingest_limit: int = int(os.getenv("CAMARA_LIMITE", "1000"))
    ingest_themes: str = os.getenv("CAMARA_TEMAS", "")


settings = Settings()
