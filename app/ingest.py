import argparse
from contextlib import closing
import io
import json
import logging
import time
from datetime import date
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader

from app import db
from app.chunking import chunk_text
from app.providers import embed_texts
from app.settings import settings
from app.tracing import flush_traces

logger = logging.getLogger(__name__)
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_API_BYTES = 2 * 1024 * 1024
MAX_REQUEST_ATTEMPTS = 6


def request_bytes(url: str, accept: str, allowed_hosts: set[str]) -> bytes:
    """Baixa resposta HTTPS de host permitido, com limite de bytes e novas tentativas."""
    parsed_url = urlparse(url)
    host = (parsed_url.hostname or "").lower()
    if parsed_url.scheme != "https":
        raise RuntimeError("A ingestão exige HTTPS para consultar a Câmara.")
    if host not in allowed_hosts:
        raise RuntimeError("A ingestão aceita somente endpoints oficiais da Câmara.")
    request = Request(url, headers={"Accept": accept, "User-Agent": "rag-legislativo/1.0"})
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            with urlopen(request, timeout=30) as response:
                final_url = urlparse(response.geturl())
                if final_url.scheme != "https" or (final_url.hostname or "").lower() not in allowed_hosts:
                    raise RuntimeError("A Câmara redirecionou a requisição para um host não permitido.")
                body = response.read(MAX_API_BYTES + 1)
                if len(body) > MAX_API_BYTES:
                    raise RuntimeError("Resposta da API da Câmara excedeu o limite de 2 MB.")
                return body
        except HTTPError as exc:
            if (exc.code != 429 and exc.code < 500) or attempt == MAX_REQUEST_ATTEMPTS - 1:
                raise RuntimeError(f"A API da Câmara respondeu HTTP {exc.code} após {attempt + 1} tentativas.") from exc
        except (URLError, TimeoutError, RemoteDisconnected) as exc:
            if attempt == MAX_REQUEST_ATTEMPTS - 1:
                raise RuntimeError(f"Falha ao consultar a Câmara após {attempt + 1} tentativas: {exc.__class__.__name__}") from exc
        delay = min(0.5 * (2 ** attempt), 8)
        logger.warning("Falha temporária na Câmara; nova tentativa %s/%s em %ss.", attempt + 2, MAX_REQUEST_ATTEMPTS, delay)
        time.sleep(delay)


def request_json(url: str) -> dict:
    """Consulta endpoint JSON oficial da Câmara e decodifica sua resposta."""
    try:
        body = request_bytes(url, "application/json", {"dadosabertos.camara.leg.br"})
        time.sleep(0.25)
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("A Câmara retornou JSON inválido.") from exc


def extract_pdf_text(url: str) -> str:
    """Extrai texto de PDF oficial da Câmara dentro dos limites de tamanho definidos."""
    host = (urlparse(url).hostname or "").lower()
    if not _is_camara_host(host):
        return ""
    try:
        parsed = urlparse(url)
        if parsed.scheme == "http":
            url = parsed._replace(scheme="https").geturl()
        request = Request(url, headers={"User-Agent": "rag-legislativo/1.0"})
        with urlopen(request, timeout=30) as response:
            final_host = (urlparse(response.geturl()).hostname or "").lower()
            if not _is_camara_host(final_host):
                return ""
            raw = response.read(MAX_PDF_BYTES + 1)
        if len(raw) > MAX_PDF_BYTES:
            logger.warning("PDF acima do limite ignorado: %s", url)
            return ""
        text = []
        chars = 0
        for page in PdfReader(io.BytesIO(raw)).pages:
            page_text = page.extract_text() or ""
            text.append(page_text[:200_000 - chars])
            chars += len(page_text)
            if chars >= 200_000:
                break
        return "\n".join(text).strip()
    except Exception as exc:
        logger.warning("Não foi possível extrair o PDF %s: %s", url, exc.__class__.__name__)
        return ""


def _is_camara_host(host: str) -> bool:
    """Verifica se o hostname pertence a um domínio oficial da Câmara."""
    return host == "camara.gov.br" or host.endswith(".camara.gov.br") or host == "camara.leg.br" or host.endswith(".camara.leg.br")


def format_document(item: dict, include_pdfs: bool = True) -> dict | None:
    """Monta o registro indexável com metadados, ementa e texto integral disponível."""
    source_id = item.get("id")
    if not source_id:
        return None
    signature = f"{item.get('siglaTipo', 'PROP')} {item.get('numero', '')}/{item.get('ano', '')}".strip()
    status = item.get("statusProposicao") or {}
    ementa = item.get("ementa") or ""
    explanation = item.get("explicacaoEmenta") or ""
    keywords = item.get("keywords") or item.get("indexacao") or ""
    last_dispatch = item.get("ultimoDespacho") or ""
    substantive = ementa or explanation or keywords or last_dispatch
    parts = [f"Proposição: {signature}", f"Ementa: {ementa}"]
    if keywords:
        parts.append(f"Palavras-chave: {keywords}")
    if explanation:
        parts.append(f"Explicação da ementa: {explanation}")
    if last_dispatch:
        parts.append(f"Último despacho: {last_dispatch}")
    current_status = status.get("descricaoTramitacao") or status.get("descricaoSituacao")
    current_status = current_status or item.get("situacao") or item.get("regimeTramitacao")
    if current_status:
        parts.append(f"Situação/tramitação atual: {current_status}")
    pdf_url = item.get("urlInteiroTeor")
    pdf_text = ""
    if include_pdfs and pdf_url:
        pdf_text = extract_pdf_text(pdf_url)
        if pdf_text:
            parts.append(f"Texto integral oficial:\n{pdf_text}")
    if not substantive and not pdf_text:
        return None
    content = "\n".join(part for part in parts if part.split(":", 1)[-1].strip())
    return {
        "document_id": str(source_id),
        "reference": signature if signature != "PROP /" else str(source_id),
        "title": signature + (f" — {ementa}" if ementa else ""),
        "url": item.get("uri") or f"{settings.camara_base_url}/proposicoes/{source_id}",
        "date": item.get("dataApresentacao"),
        "content": content,
    }


def ingest(limit: int, start: str, end: str, themes: list[str], include_pdfs: bool = True) -> dict:
    """Pagina proposições, gera embeddings e persiste documentos e chunks no banco."""
    if not 1 <= limit <= 5000:
        raise ValueError("CAMARA_LIMITE deve estar entre 1 e 5000.")
    date.fromisoformat(start)
    if end:
        date.fromisoformat(end)
    imported = skipped = examined = 0
    page = 1
    with closing(db.connect()) as conn:
        while imported < limit and examined < limit * 5:
            params = {
            "dataApresentacaoInicio": start,
            "itens": 100,
            "pagina": page,
            "ordenarPor": "id",
            "ordem": "ASC",
            }
            if end:
                params["dataApresentacaoFim"] = end
            if themes:
                params["codTema"] = ",".join(themes)
            result = request_json(f"{settings.camara_base_url}/proposicoes?{urlencode(params)}")
            items = result.get("dados", [])
            if not items:
                break
            for item in items:
                examined += 1
                if include_pdfs and item.get("uri") and not item.get("urlInteiroTeor"):
                    try:
                        details = request_json(item["uri"]).get("dados", {})
                        if isinstance(details, dict):
                            item = item | details
                    except RuntimeError as exc:
                        logger.warning("Detalhes da proposição %s indisponíveis; usando dados da listagem: %s", item.get("id"), exc)
                document = format_document(item, include_pdfs)
                if not document:
                    skipped += 1
                    continue
                chunks = chunk_text(document["document_id"], document["content"])
                if not chunks:
                    skipped += 1
                    continue
                vectors = []
                for offset in range(0, len(chunks), 64):
                    vectors.extend(embed_texts([chunk["text"] for chunk in chunks[offset:offset + 64]]))
                if document["date"]:
                    document["date"] = document["date"][:10]
                db.save_document(conn, document, chunks, vectors)
                imported += 1
                if imported % 50 == 0:
                    logger.info("Indexadas %s proposições", imported)
                if imported >= limit:
                    break
            page += 1
    return {"indexed": imported, "skipped": skipped, "examined": examined}


def main() -> None:
    """Lê opções da linha de comando e executa a ingestão do corpus legislativo."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Indexa proposições oficiais da Câmara em pgvector.")
    parser.add_argument("--limit", type=int, default=settings.ingest_limit)
    parser.add_argument("--start", default=settings.ingest_start)
    parser.add_argument("--end", default=settings.ingest_end)
    parser.add_argument("--themes", default=settings.ingest_themes, help="Códigos de tema separados por vírgula")
    parser.add_argument("--ementas-only", action="store_true", help="Não baixar o texto integral em PDF")
    args = parser.parse_args()
    db.initialize_schema()
    try:
        summary = ingest(args.limit, args.start, args.end, [x.strip() for x in args.themes.split(",") if x.strip()], not args.ementas_only)
    finally:
        flush_traces()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
