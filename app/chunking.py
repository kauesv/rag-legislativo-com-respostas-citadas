import hashlib

from app.settings import settings


def chunk_text(document_id: str, text: str) -> list[dict]:
    """Normaliza o texto e o divide em trechos sobrepostos com IDs determinísticos."""
    if settings.chunk_size < 1 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("CHUNK_SIZE deve ser positivo e CHUNK_OVERLAP deve ser menor que ele.")
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + settings.chunk_size, len(cleaned))
        body = cleaned[start:end].strip()
        if body:
            chunk_id = hashlib.sha256(f"{document_id}:{start}:{body}".encode()).hexdigest()[:24]
            chunks.append({"chunk_id": chunk_id, "text": body})
        if end == len(cleaned):
            break
        start = end - settings.chunk_overlap
    return chunks
