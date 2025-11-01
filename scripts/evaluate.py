"""Evaluation pipeline for recall quality and latency metrics.

Usage:
    python scripts/evaluate.py --config configs/eval.toml

Outputs per run are stored in an artifacts directory defined in the
configuration. The script computes classical information retrieval metrics
using the manually labelled `data/test.csv` file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import tomllib

# Ensure repository root is on sys.path for module imports
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bm25_retrieval import BM25Retriever, Candidates, load_food_candidates


# ---------------------------------------------------------------------------
# Data structures


@dataclass
class QueryRecord:
    query_id: str
    text: str
    relevance: Dict[str, float]


@dataclass
class RetrievalResult:
    query_id: str
    latency_ms: float
    retrieved: Sequence[Candidates]
    scores: Sequence[float]


# ---------------------------------------------------------------------------
# Metric helpers


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k == 0:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k]):
        if rel == 0:
            continue
        dcg += (2**rel - 1) / math.log2(idx + 2)
    return dcg


def ndcg_at_k(retrieved_ids: Sequence[str], relevances: Dict[str, float], k: int) -> float:
    gains = [relevances.get(doc_id, 0.0) for doc_id in retrieved_ids]
    ideal = sorted(relevances.values(), reverse=True)
    actual_dcg = dcg_at_k(gains, k)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def average_precision(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / idx
    if hits == 0:
        return 0.0
    return precision_sum / len(relevant_set)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant_set = set(relevant_ids)
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / idx
    return 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[int(idx)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (idx - lower)


# ---------------------------------------------------------------------------
# Utility helpers


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_test_data(test_path: Path, min_relevance: int) -> Dict[str, QueryRecord]:
    queries: Dict[str, QueryRecord] = {}
    with test_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row["query_id"]
            text = row["query_text"]
            doc_id = row["item_id"]
            relevance = float(row.get("score", 0) or 0)
            if query_id not in queries:
                queries[query_id] = QueryRecord(query_id=query_id, text=text, relevance={})
            if relevance >= min_relevance:
                queries[query_id].relevance[doc_id] = relevance
    return queries


def ensure_output_dir(base_dir: Path, tag: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Evaluation core


def evaluate(config_path: Path) -> None:
    config = load_config(config_path)

    retriever_cfg = config["retriever"]
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    output_cfg = config["output"]

    ks: List[int] = sorted(set(eval_cfg.get("ks", [10])))
    max_k = max(ks)
    metrics_requested = set(eval_cfg.get("metrics", ["precision", "recall", "ndcg"]))
    min_rel = int(eval_cfg.get("min_relevance", 1))

    test_path = Path(data_cfg["test_path"])
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # Load evaluation data
    query_records = load_test_data(test_path, min_rel)
    if not query_records:
        raise ValueError("No queries with relevance >= min_relevance found in test data.")

    # Initialise retriever
    items_path = Path(retriever_cfg["items_path"])
    params = retriever_cfg.get("params", {})
    documents = load_food_candidates(items_path)
    retriever = BM25Retriever(documents, **params)

    # Perform retrieval for each query
    retrievals: List[RetrievalResult] = []
    evaluated_queries = [qr for qr in query_records.values() if qr.relevance]
    skipped_queries = [qr.query_id for qr in query_records.values() if not qr.relevance]

    if not evaluated_queries:
        raise ValueError("No queries contain positive relevance labels; cannot compute metrics.")

    for qr in evaluated_queries:
        start = time.perf_counter()
        results = retriever.search(qr.text, top_k=max_k)
        latency_ms = (time.perf_counter() - start) * 1000
        docs, scores = zip(*results) if results else ([], [])
        retrievals.append(RetrievalResult(qr.query_id, latency_ms, docs, scores))

    # Compute metrics per query
    per_query_rows: List[dict] = []
    aggregated: Dict[str, List[float]] = defaultdict(list)
    coverage_counts: Dict[int, int] = {k: 0 for k in ks}

    for result in retrievals:
        qr = query_records[result.query_id]
        doc_ids = [doc.id for doc in result.retrieved]
        rel_ids = qr.relevance.keys()

        row = {
            "query_id": result.query_id,
            "query_text": qr.text,
            "latency_ms": result.latency_ms,
        }

        for k in ks:
            prefix = f"@{k}"
            if "precision" in metrics_requested:
                value = precision_at_k(doc_ids, rel_ids, k)
                row[f"precision{prefix}"] = value
                aggregated[f"precision{prefix}"].append(value)
            if "recall" in metrics_requested:
                value = recall_at_k(doc_ids, rel_ids, k)
                row[f"recall{prefix}"] = value
                aggregated[f"recall{prefix}"].append(value)
            if "ndcg" in metrics_requested:
                value = ndcg_at_k(doc_ids, qr.relevance, k)
                row[f"ndcg{prefix}"] = value
                aggregated[f"ndcg{prefix}"].append(value)
            if any(doc_id in qr.relevance for doc_id in doc_ids[:k]):
                coverage_counts[k] += 1

        if "map" in metrics_requested:
            ap = average_precision(doc_ids, rel_ids)
            row["average_precision"] = ap
            aggregated["average_precision"].append(ap)
        if "mrr" in metrics_requested:
            rr = reciprocal_rank(doc_ids, rel_ids)
            row["reciprocal_rank"] = rr
            aggregated["reciprocal_rank"].append(rr)

        per_query_rows.append(row)

    latencies = [res.latency_ms for res in retrievals]

    # Aggregate summary
    summary_metrics = {}
    for key, values in aggregated.items():
        summary_metrics[key] = sum(values) / len(values)

    coverage_summary = {f"coverage@{k}": coverage_counts[k] / len(retrievals) for k in ks}

    timing_stats = {
        "query_count": len(retrievals),
        "skipped_query_count": len(skipped_queries),
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }

    summary = {
        "config": deepcopy(config),
        "query_count": len(retrievals),
        "skipped_queries": skipped_queries,
        "metrics": summary_metrics,
        "coverage": coverage_summary,
        "timing": timing_stats,
    }

    # Persist artifacts
    output_dir = ensure_output_dir(Path(output_cfg["dir"]), output_cfg.get("tag", "run"))
    save_json(output_dir / "metrics_summary.json", summary)
    save_csv(output_dir / "per_query_metrics.csv", per_query_rows)
    save_json(output_dir / "timing.json", timing_stats)

    # Copy config for reproducibility
    with (output_dir / "config_used.toml").open("wb") as f:
        f.write(config_path.read_bytes())

    print(f"Evaluation completed. Artifacts written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recall pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval.toml"),
        help="Path to the evaluation configuration file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config)

