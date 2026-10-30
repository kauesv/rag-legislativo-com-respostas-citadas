import argparse
import json
import math
import re
import statistics
from pathlib import Path
from time import perf_counter

from app.providers import generate_answer
from app.retrieval import retrieve
from app.settings import settings
from app.tracing import log_evaluation

GOLDEN_SET = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "golden.jsonl"


def recall_at_k(ranked: list[str], expected: set[str]) -> float:
    """Calcula a fração de referências esperadas presentes entre os resultados."""
    return len(set(ranked) & expected) / len(expected) if expected else math.nan


def ndcg_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """Calcula NDCG@K para relevância binária das referências esperadas."""
    if not expected:
        return math.nan
    dcg = sum(1 / math.log2(rank + 2) for rank, doc_id in enumerate(ranked[:k]) if doc_id in expected)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(expected))))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], percentile_value: int) -> float:
    """Retorna o percentil solicitado, usando o único valor quando há uma amostra."""
    return statistics.quantiles(values, n=100)[percentile_value - 1] if len(values) > 1 else values[0]


def groundedness_proxy(answer: str, cited_texts: list[str]) -> float:
    """Estima cobertura lexical da resposta nos trechos citados; não mede entailment."""
    terms = set(re.findall(r"[\wÀ-ÿ]{3,}", answer.casefold()))
    if not terms:
        return 0.0
    evidence = set(re.findall(r"[\wÀ-ÿ]{3,}", " ".join(cited_texts).casefold()))
    return len(terms & evidence) / len(terms)


def evaluate_strategy(strategy: str, questions: list[dict], include_answers: bool) -> dict:
    """Avalia recuperação e, opcionalmente, respostas, citações e abstenções."""
    recalls, ndcgs, retrieval_times, generation_times = [], [], [], []
    valid_citations = abstention_correct = answer_count = 0
    groundedness_scores, input_tokens, output_tokens = [], [], []
    ambiguous_count = ambiguous_covered = 0
    for item in questions:
        start = perf_counter()
        chunks = retrieve(item["question"], strategy, settings.max_top_k)
        retrieval_ms = (perf_counter() - start) * 1000
        retrieval_times.append(retrieval_ms)
        expected = set(item["expected_references"])
        ranked = list(dict.fromkeys(chunk["reference"] for chunk in chunks))
        if expected:
            recalls.append(recall_at_k(ranked, expected))
            ndcgs.append(ndcg_at_k(ranked, expected, settings.max_top_k))

        if include_answers:
            answer_count += 1
            generation_ms = 0.0
            generated = {"answer": "", "citations": []}
            if chunks:
                start = perf_counter()
                generated = generate_answer(item["question"], chunks)
                generation_ms = (perf_counter() - start) * 1000
            generation_times.append(generation_ms)
            retrieved_ids = {chunk["chunk_id"] for chunk in chunks}
            citations = generated["citations"]
            valid = bool(citations) and all(citation in retrieved_ids for citation in citations)
            abstained = not generated["answer"] or not valid
            valid_citations += int(valid)
            abstention_correct += int(abstained == (not expected))
            cited_texts = [chunk["text"] for chunk in chunks if chunk["chunk_id"] in citations]
            if generated["answer"]:
                groundedness_scores.append(groundedness_proxy(generated["answer"], cited_texts))
            usage = generated.get("usage", {})
            input_tokens.append(usage.get("input", 0))
            output_tokens.append(usage.get("output", 0))
            if item["label"] == "ambiguous":
                ambiguous_count += 1
                cited_refs = {chunk["reference"] for chunk in chunks if chunk["chunk_id"] in citations}
                ambiguous_covered += int(bool(valid) and expected.issubset(cited_refs))

    metrics = {
        "recall_at_k": statistics.mean(recalls) if recalls else None,
        "ndcg_at_k": statistics.mean(ndcgs) if ndcgs else None,
        "retrieval_p50_ms": statistics.median(retrieval_times),
        "retrieval_p95_ms": percentile(retrieval_times, 95),
    }
    if include_answers:
        metrics.update({
            "citation_valid_rate": valid_citations / answer_count,
            "groundedness_proxy": statistics.mean(groundedness_scores) if groundedness_scores else 0.0,
            "input_tokens_total": sum(input_tokens),
            "output_tokens_total": sum(output_tokens),
            "abstention_correct_rate": abstention_correct / answer_count,
            "ambiguous_reference_coverage": ambiguous_covered / ambiguous_count if ambiguous_count else None,
            "generation_p50_ms": statistics.median(generation_times),
            "generation_p95_ms": percentile(generation_times, 95),
        })
    log_evaluation(strategy, metrics, len(questions), include_answers)
    return metrics


def main() -> None:
    """Carrega o golden set, executa as estratégias e imprime métricas agregadas."""
    parser = argparse.ArgumentParser(description="Compara BM25 e embeddings no golden set versionado.")
    parser.add_argument("--answers", action="store_true", help="Também chama o LLM para medir citações/abstenção (pode ter custo)")
    args = parser.parse_args()
    questions = [json.loads(line) for line in GOLDEN_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 50 <= len(questions) <= 100:
        raise ValueError(f"O golden set precisa conter 50–100 perguntas; encontrou {len(questions)}.")
    result = {strategy: evaluate_strategy(strategy, questions, args.answers) for strategy in ("bm25", "embedding")}
    print(json.dumps({"golden_set": "v1", "questions": len(questions), "metrics": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
