"""Experiment 2: Vector Embedding Model Comparison

This script runs experiments comparing different embedding models and dimensions
on retrieval performance, latency, cost, and storage.

Usage:
    python3 scripts/eval_vector.py

Outputs are stored in artifacts/eval_vector_runs/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Ensure repository root is on sys.path for module imports
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.bm25_retrieval import Candidates, load_food_candidates
from src.vector_retrieval import VectorRetriever, estimate_batch_tokens


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryRecord:
    query_id: str
    text: str
    relevance: Dict[str, float]  # Filtered relevance (for Precision/Recall)
    relevance_all: Dict[str, float]  # All relevance scores including 0 (for NDCG)


@dataclass
class RetrievalResult:
    query_id: str
    latency_ms: float
    retrieved: Sequence[Candidates]
    scores: Sequence[float]


@dataclass
class ExperimentConfig:
    model_name: str
    dimensions: int | None
    api_base: str | None = None
    api_key: str | None = None
    max_tokens_per_request: int = 8192
    max_items_per_batch: int | None = None
    rpm_limit: int = 500
    timeout: float = 120.0
    normalize_embeddings: bool = True


@dataclass
class CostMetrics:
    total_requests: int = 0
    requests_per_batch: List[int] = field(default_factory=list)
    tokens_per_request: List[int] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0  # For embeddings, this is 0
    storage_size_bytes: int = 0  # Total storage for all embeddings


@dataclass
class ExperimentResults:
    config: ExperimentConfig
    query_count: int
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_stats: Dict[str, float] = field(default_factory=dict)
    cost: CostMetrics = field(default_factory=CostMetrics)
    retrieval_results: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric helpers (reused from evaluate.py)
# ---------------------------------------------------------------------------

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


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant_set = set(relevant_ids)
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / idx
    return 0.0


def coverage_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    hits_in_topk = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_set)
    return 1.0 if hits_in_topk > 0 else 0.0


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
# ---------------------------------------------------------------------------

def load_test_data(
    test_path: Path,
    min_relevance: float,
    min_relevance_ndcg: float = 0.0,
    queries_path: Path | None = None
) -> Dict[str, QueryRecord]:
    """Load test data from CSV file(s)."""
    queries: Dict[str, QueryRecord] = {}
    
    if queries_path and queries_path.exists():
        # Load query texts first
        query_texts: Dict[str, str] = {}
        with queries_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = row.get("id") or row.get("query_id")
                text = row.get("search_term_pt") or row.get("query") or row.get("text") or row.get("query_text")
                if query_id and text:
                    query_texts[query_id] = text
        
        # Load scores
        with test_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = row["query_id"]
                doc_id = row["item_id"]
                score_str = row.get("score", "").strip()
                
                if not score_str:
                    continue
                    
                try:
                    relevance = float(score_str)
                except ValueError:
                    continue
                
                text = query_texts.get(query_id, "")
                if not text:
                    continue
                
                if query_id not in queries:
                    queries[query_id] = QueryRecord(
                        query_id=query_id,
                        text=text,
                        relevance={},
                        relevance_all={}
                    )
                
                if relevance >= min_relevance_ndcg:
                    queries[query_id].relevance_all[doc_id] = relevance
                
                if relevance >= min_relevance:
                    queries[query_id].relevance[doc_id] = relevance
    
    return queries


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


def ensure_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"eval_vector_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def calculate_storage_size(embedding_dim: int, num_items: int, num_queries: int) -> int:
    """Calculate storage size in bytes for embeddings.
    
    Each float32 value is 4 bytes.
    Storage includes: document embeddings + query embeddings (for retrieval).
    """
    # Document embeddings: num_items * dim * 4 bytes
    doc_storage = num_items * embedding_dim * 4
    # Query embeddings (for retrieval): num_queries * dim * 4 bytes
    query_storage = num_queries * embedding_dim * 4
    return doc_storage + query_storage


# ---------------------------------------------------------------------------
# Experiment core
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    candidates: List[Candidates],
    query_records: Dict[str, QueryRecord],
    ks: List[int] = [5, 10],
    min_relevance: float = 1.0,
) -> ExperimentResults:
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Running experiment: {config.model_name}")
    if config.dimensions:
        print(f"  Dimensions: {config.dimensions}")
    else:
        print(f"  Dimensions: default (model-dependent)")
    print(f"{'='*70}")
    
    # Initialize retriever
    print("Initializing vector retriever...")
    retriever = VectorRetriever(
        candidates,
        model_name=config.model_name,
        api_base=config.api_base,
        api_key=config.api_key,
        max_tokens_per_request=config.max_tokens_per_request,
        max_items_per_batch=config.max_items_per_batch,
        rpm_limit=config.rpm_limit,
        timeout=config.timeout,
        dimensions=config.dimensions,
        normalize_embeddings=config.normalize_embeddings,
    )
    
    # Track cost metrics
    cost = CostMetrics()
    
    # Estimate token usage for document embeddings
    # Note: VectorRetriever already computed embeddings during initialization,
    # but we estimate tokens here for cost tracking
    doc_texts = [doc.text for doc in candidates]
    estimated_doc_tokens = estimate_batch_tokens(doc_texts)
    cost.total_input_tokens += estimated_doc_tokens
    
    # Estimate number of API requests for document embeddings
    # VectorRetriever uses batch_by_tokens, so we estimate batch count
    from src.vector_retrieval import batch_by_tokens
    doc_batches = list(batch_by_tokens(
        doc_texts,
        max_tokens_per_request=config.max_tokens_per_request,
        max_items_per_batch=config.max_items_per_batch,
        model_name=config.model_name,
    ))
    cost.total_requests += len(doc_batches)
    # Estimate tokens per batch
    for batch in doc_batches:
        batch_tokens = estimate_batch_tokens(batch)
        cost.tokens_per_request.append(batch_tokens)
        cost.requests_per_batch.append(len(batch))
    
    # Determine embedding dimension (from config or model default)
    embedding_dim = config.dimensions
    if embedding_dim is None:
        # Use model defaults
        if "small" in config.model_name:
            embedding_dim = 1536
        elif "large" in config.model_name:
            embedding_dim = 3072
        else:
            embedding_dim = 1536  # fallback
    
    # Perform retrieval for each query
    retrievals: List[RetrievalResult] = []
    evaluated_queries = [qr for qr in query_records.values() if qr.relevance]
    
    print(f"\nPerforming retrieval for {len(evaluated_queries)} queries...")
    max_k = max(ks)
    
    for idx, qr in enumerate(evaluated_queries, 1):
        print(f"  Processing query {idx}/{len(evaluated_queries)}: {qr.query_id}")
        
        # Estimate query tokens (each query is a separate API request)
        query_tokens = estimate_batch_tokens([qr.text])
        cost.total_input_tokens += query_tokens
        cost.total_requests += 1  # Each query is one API request
        cost.tokens_per_request.append(query_tokens)
        cost.requests_per_batch.append(1)
        
        start = time.perf_counter()
        results = retriever.search(qr.text, top_k=max_k)
        latency_ms = (time.perf_counter() - start) * 1000
        
        docs, scores = zip(*results) if results else ([], [])
        retrievals.append(RetrievalResult(qr.query_id, latency_ms, docs, scores))
    
    # Calculate storage cost
    cost.storage_size_bytes = calculate_storage_size(
        embedding_dim, len(candidates), len(evaluated_queries)
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics: Dict[str, float] = defaultdict(list)
    latency_values: List[float] = []
    retrieval_results: List[Dict] = []
    
    for result in retrievals:
        qr = query_records[result.query_id]
        doc_ids = [doc.id for doc in result.retrieved]
        rel_ids = qr.relevance.keys()
        
        latency_values.append(result.latency_ms)
        
        # Store retrieval results
        retrieval_entry = {
            "query_id": result.query_id,
            "query_text": qr.text,
            "retrieved_items": [
                {
                    "item_id": doc.id,
                    "item_name": doc.name,
                    "rank": rank + 1,
                    "score": float(score)
                }
                for rank, (doc, score) in enumerate(zip(result.retrieved, result.scores))
            ]
        }
        retrieval_results.append(retrieval_entry)
        
        # Calculate metrics for each k
        for k in ks:
            # Precision@K
            value = precision_at_k(doc_ids, rel_ids, k)
            metrics[f"precision@{k}"].append(value)
            
            # Recall@K
            value = recall_at_k(doc_ids, rel_ids, k)
            metrics[f"recall@{k}"].append(value)
            
            # NDCG@K
            value = ndcg_at_k(doc_ids, qr.relevance_all, k)
            metrics[f"ndcg@{k}"].append(value)
            
            # Coverage@K
            value = coverage_at_k(doc_ids, rel_ids, k)
            metrics[f"coverage@{k}"].append(value)
        
        # MRR (only needs to be calculated once per query)
        value = reciprocal_rank(doc_ids, rel_ids)
        metrics["mrr"].append(value)
    
    # Aggregate metrics
    aggregated_metrics = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in metrics.items()
    }
    
    # Calculate latency statistics
    latency_stats = {
        "avg_latency_ms": statistics.mean(latency_values) if latency_values else 0.0,
        "median_latency_ms": statistics.median(latency_values) if latency_values else 0.0,
        "p95_latency_ms": percentile(latency_values, 95),
        "p99_latency_ms": percentile(latency_values, 99),
        "min_latency_ms": min(latency_values) if latency_values else 0.0,
        "max_latency_ms": max(latency_values) if latency_values else 0.0,
    }
    
    return ExperimentResults(
        config=config,
        query_count=len(evaluated_queries),
        metrics=aggregated_metrics,
        latency_stats=latency_stats,
        cost=cost,
        retrieval_results=retrieval_results,
    )


def main():
    """Main experiment runner."""
    # TODO: set experiment
    # Define experiment configurations
    experiment_configs = [
        # text-embedding-3-small configurations
        ExperimentConfig(model_name="text-embedding-3-small", dimensions=1536),
        ExperimentConfig(model_name="text-embedding-3-small", dimensions=768),
        # text-embedding-3-large configurations
        ExperimentConfig(model_name="text-embedding-3-large", dimensions=3072),
        ExperimentConfig(model_name="text-embedding-3-large", dimensions=1536),
        ExperimentConfig(model_name="text-embedding-3-large", dimensions=768),
    ]
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("=" * 70)
        print("ERROR: OPENAI_API_KEY not found!")
        print("=" * 70)
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        print("  or create a .env file with: OPENAI_API_KEY=sk-your-key-here")
        print("=" * 70)
        return
    
    # Set API key for all configs
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    if api_base.strip() == "":
        api_base = "https://api.openai.com/v1"
    
    for config in experiment_configs:
        config.api_key = api_key
        config.api_base = api_base
    
    # Load data
    data_dir = REPO_ROOT / "data" / "test"
    queries_path = data_dir / "10_queries.csv"
    items_path = data_dir / "500_items.csv"
    test_path = data_dir / "test_query_new.csv"
    
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    print("Loading data...")
    candidates = load_food_candidates(items_path)
    print(f"  Loaded {len(candidates)} items")

    # TODO: set experiment
    query_records = load_test_data(
        test_path,
        min_relevance=6.0,
        min_relevance_ndcg=0.0,
        queries_path=queries_path,
    )
    print(f"  Loaded {len(query_records)} queries")
    
    # Run experiments
    all_results: List[ExperimentResults] = []
    ks = [5, 10]
    
    for config in experiment_configs:
        try:
            results = run_experiment(
                config,
                candidates,
                query_records,
                ks=ks,
                min_relevance=1.0,
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nERROR running experiment {config.model_name} (dim={config.dimensions}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate report
    print(f"\n{'='*70}")
    print("Generating experiment report...")
    print(f"{'='*70}")
    
    output_dir = ensure_output_dir(REPO_ROOT / "artifacts" / "eval_vector_runs")
    
    # Create summary report
    summary = {
        "experiment_name": "Experiment 2: Vector Embedding Model Comparison",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {
            "queries": str(queries_path),
            "items": str(items_path),
            "test_set": str(test_path),
        },
        "metrics_evaluated": ["precision@5", "recall@5", "ndcg@5", "coverage@5", "mrr",
                              "precision@10", "recall@10", "ndcg@10", "coverage@10"],
        "configurations": [],
        "results": [],
    }
    
    # Prepare detailed results
    comparison_rows = []
    
    for results in all_results:
        config_dict = {
            "model_name": results.config.model_name,
            "dimensions": results.config.dimensions,
            "normalize_embeddings": results.config.normalize_embeddings,
        }
        
        avg_tokens = sum(results.cost.tokens_per_request) / len(results.cost.tokens_per_request) if results.cost.tokens_per_request else 0
        result_dict = {
            "config": config_dict,
            "query_count": results.query_count,
            "metrics": results.metrics,
            "latency": results.latency_stats,
            "cost": {
                "total_requests": results.cost.total_requests,
                "total_input_tokens": results.cost.total_input_tokens,
                "avg_tokens_per_request": avg_tokens,
                "max_tokens_per_request": max(results.cost.tokens_per_request) if results.cost.tokens_per_request else 0,
                "min_tokens_per_request": min(results.cost.tokens_per_request) if results.cost.tokens_per_request else 0,
                "storage_size_bytes": results.cost.storage_size_bytes,
                "storage_size_mb": results.cost.storage_size_bytes / (1024 * 1024),
            },
        }
        
        summary["configurations"].append(config_dict)
        summary["results"].append(result_dict)
        
        # Create comparison row
        row = {
            "model_name": results.config.model_name,
            "dimensions": results.config.dimensions or "default",
            **{f"metric_{k}": v for k, v in results.metrics.items()},
            **{f"latency_{k}": v for k, v in results.latency_stats.items()},
            "total_requests": results.cost.total_requests,
            "total_input_tokens": results.cost.total_input_tokens,
            "avg_tokens_per_request": sum(results.cost.tokens_per_request) / len(results.cost.tokens_per_request) if results.cost.tokens_per_request else 0,
            "storage_size_mb": results.cost.storage_size_bytes / (1024 * 1024),
        }
        comparison_rows.append(row)
        
        # Save individual experiment results
        exp_name = f"{results.config.model_name}_dim{results.config.dimensions or 'default'}"
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        save_json(exp_dir / "config.json", config_dict)
        save_json(exp_dir / "metrics.json", {
            "metrics": results.metrics,
            "latency": results.latency_stats,
            "cost": {
                "total_requests": results.cost.total_requests,
                "total_input_tokens": results.cost.total_input_tokens,
                "avg_tokens_per_request": sum(results.cost.tokens_per_request) / len(results.cost.tokens_per_request) if results.cost.tokens_per_request else 0,
                "max_tokens_per_request": max(results.cost.tokens_per_request) if results.cost.tokens_per_request else 0,
                "min_tokens_per_request": min(results.cost.tokens_per_request) if results.cost.tokens_per_request else 0,
                "storage_size_bytes": results.cost.storage_size_bytes,
                "storage_size_mb": results.cost.storage_size_bytes / (1024 * 1024),
            },
        })
        save_json(exp_dir / "retrieval_results.json", results.retrieval_results)
    
    # Save summary
    save_json(output_dir / "experiment_summary.json", summary)
    save_csv(output_dir / "comparison.csv", comparison_rows)
    
    print(f"\nExperiment completed!")
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary saved to: {output_dir / 'experiment_summary.json'}")
    print(f"Comparison CSV saved to: {output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()

