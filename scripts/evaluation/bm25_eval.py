#!/usr/bin/env python3
"""BM25 retrieval evaluation script.

This script:
1. Loads queries from data/test/30_queries.csv
2. Performs BM25 search on data/raw/5k_items.csv (using cached BM25 index)
3. Evaluates results against LLM-scored test set data/test/test_query.csv
4. Generates comprehensive evaluation report

Evaluation metrics:
- Precision@K, Recall@K, NDCG@K, MRR, Coverage@K
- Average/Max/Min latency
- Per-query results and ground truth comparison
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path for module imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from src.bm25_retrieval import BM25Retriever, Candidates, load_food_candidates


def load_queries(queries_path: Path) -> Dict[str, str]:
    """Load queries from CSV file.
    
    Returns:
        Dictionary mapping query_id to query_text.
    """
    queries = {}
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    with queries_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("id") or row.get("query_id") or ""
            query_text = row.get("search_term_pt") or row.get("query") or row.get("text") or ""
            if query_id and query_text.strip():
                queries[query_id] = query_text.strip()
    
    return queries


def load_ground_truth(test_query_path: Path) -> Dict[str, Dict[str, float]]:
    """Load ground truth scores from test_query.csv.
    
    Returns:
        Dictionary mapping query_id -> {item_id: score}
        Items not in test set are implicitly scored as 0.0
    """
    ground_truth = {}
    
    if not test_query_path.exists():
        raise FileNotFoundError(f"Test query file not found: {test_query_path}")
    
    with test_query_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("query_id", "").strip()
            item_id = row.get("item_id", "").strip()
            score_str = row.get("score", "").strip()
            
            if not query_id or not item_id:
                continue
            
            try:
                score = float(score_str) if score_str else 0.0
            except (ValueError, TypeError):
                score = 0.0
            
            if query_id not in ground_truth:
                ground_truth[query_id] = {}
            ground_truth[query_id][item_id] = score
    
    return ground_truth


def get_item_score(query_id: str, item_id: str, ground_truth: Dict[str, Dict[str, float]]) -> float:
    """Get ground truth score for an item. Returns 0.0 if not in test set."""
    if query_id not in ground_truth:
        return 0.0
    return ground_truth[query_id].get(item_id, 0.0)


def precision_at_k(retrieved_items: List[str], ground_truth_items: List[str], k: int) -> float:
    """Calculate Precision@K.
    
    Args:
        retrieved_items: List of retrieved item IDs (top-K)
        ground_truth_items: List of relevant item IDs (score > 0)
        k: Cutoff position
        
    Returns:
        Precision@K value
    """
    if k == 0:
        return 0.0
    
    retrieved_k = retrieved_items[:k]
    relevant_retrieved = sum(1 for item_id in retrieved_k if item_id in ground_truth_items)
    
    return relevant_retrieved / k


def recall_at_k(retrieved_items: List[str], ground_truth_items: List[str], k: int) -> float:
    """Calculate Recall@K.
    
    Args:
        retrieved_items: List of retrieved item IDs (top-K)
        ground_truth_items: List of relevant item IDs (score > 0)
        k: Cutoff position
        
    Returns:
        Recall@K value
    """
    if len(ground_truth_items) == 0:
        return 0.0
    
    retrieved_k = retrieved_items[:k]
    relevant_retrieved = sum(1 for item_id in retrieved_k if item_id in ground_truth_items)
    
    return relevant_retrieved / len(ground_truth_items)


def dcg_at_k(scores: List[float], k: int) -> float:
    """Calculate DCG@K (Discounted Cumulative Gain).
    
    Uses standard DCG formula: score / log2(rank + 1)
    For rank 1: score / log2(2) = score / 1.0
    For rank 2: score / log2(3) ≈ score / 1.585
    etc.
    
    Args:
        scores: List of relevance scores for retrieved items
        k: Cutoff position
        
    Returns:
        DCG@K value
    """
    import math
    dcg = 0.0
    for i in range(min(k, len(scores))):
        rank = i + 1  # rank starts from 1
        dcg += scores[i] / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved_items: List[str], retrieved_scores: List[float], 
               ground_truth: Dict[str, float], k: int) -> float:
    """Calculate NDCG@K (Normalized Discounted Cumulative Gain).
    
    Args:
        retrieved_items: List of retrieved item IDs (top-K)
        retrieved_scores: List of retrieval scores (for ranking)
        ground_truth: Dictionary mapping item_id to relevance score
        k: Cutoff position
        
    Returns:
        NDCG@K value
    """
    if k == 0:
        return 0.0
    
    # Get relevance scores for retrieved items
    retrieved_k = retrieved_items[:k]
    relevance_scores = [ground_truth.get(item_id, 0.0) for item_id in retrieved_k]
    
    # Calculate DCG@K
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate IDCG@K (ideal DCG)
    ideal_scores = sorted([score for score in ground_truth.values() if score > 0], reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def mrr(retrieved_items: List[str], ground_truth_items: List[str]) -> float:
    """Calculate MRR (Mean Reciprocal Rank).
    
    Args:
        retrieved_items: List of retrieved item IDs
        ground_truth_items: List of relevant item IDs (score > 0)
        
    Returns:
        MRR value
    """
    if len(ground_truth_items) == 0:
        return 0.0
    
    for rank, item_id in enumerate(retrieved_items, start=1):
        if item_id in ground_truth_items:
            return 1.0 / rank
    
    return 0.0


def coverage_at_k(retrieved_items: List[str], ground_truth_items: List[str], k: int) -> float:
    """Calculate Coverage@K (percentage of relevant items found in top-K).
    
    Args:
        retrieved_items: List of retrieved item IDs (top-K)
        ground_truth_items: List of relevant item IDs (score > 0)
        k: Cutoff position
        
    Returns:
        Coverage@K value (percentage)
    """
    if len(ground_truth_items) == 0:
        return 0.0
    
    retrieved_k = retrieved_items[:k]
    retrieved_set = set(retrieved_k)
    ground_truth_set = set(ground_truth_items)
    
    covered = len(retrieved_set & ground_truth_set)
    return covered / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0


def evaluate_query(
    query_id: str,
    query_text: str,
    retrieved_results: List[Tuple[Candidates, float]],
    ground_truth: Dict[str, float],
    k_values: List[int] = [5, 10]
) -> Dict:
    """Evaluate a single query.
    
    Returns:
        Dictionary with evaluation metrics and results
    """
    # Extract retrieved item IDs and scores
    retrieved_items = [candidate.id for candidate, _ in retrieved_results]
    retrieved_scores = [score for _, score in retrieved_results]
    
    # Get relevant items (score > 0) from ground truth
    # Items not in test set are implicitly scored as 0.0
    ground_truth_items = [item_id for item_id, score in ground_truth.items() if score > 0]
    
    # Calculate metrics for each K
    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(retrieved_items, ground_truth_items, k)
        metrics[f'recall@{k}'] = recall_at_k(retrieved_items, ground_truth_items, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_items, retrieved_scores, ground_truth, k)
        metrics[f'coverage@{k}'] = coverage_at_k(retrieved_items, ground_truth_items, k)
    
    metrics['mrr'] = mrr(retrieved_items, ground_truth_items)
    
    # Get top 10 ground truth items sorted by score
    top_gt_items = sorted(
        [(item_id, score) for item_id, score in ground_truth.items() if score > 0],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return {
        'query_id': query_id,
        'query_text': query_text,
        'retrieved_items': retrieved_items[:10],  # Top 10
        'retrieved_results': [
            (candidate.id, candidate.name, score) 
            for candidate, score in retrieved_results[:10]
        ],
        'ground_truth_items': top_gt_items,
        'ground_truth_dict': ground_truth,  # Full ground truth dict for score lookup
        'metrics': metrics,
    }


def get_system_info() -> Dict[str, str]:
    """Get system hardware and software information."""
    import platform
    import sys
    
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': sys.version.split()[0],
        'machine': platform.machine(),
        'architecture': platform.architecture()[0],
    }
    
    # Try to get CPU info
    try:
        import psutil
        info['cpu_count'] = str(psutil.cpu_count())
        info['cpu_freq'] = f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "N/A"
        info['memory_total'] = f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    except ImportError:
        info['cpu_count'] = "N/A (install psutil for details)"
        info['cpu_freq'] = "N/A"
        info['memory_total'] = "N/A"
    
    # Try to detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['gpu'] = "Apple Metal (MPS)"
            info['gpu_memory'] = "N/A"
        else:
            info['gpu'] = "None (CPU only)"
            info['gpu_memory'] = "N/A"
    except ImportError:
        info['gpu'] = "N/A (PyTorch not installed)"
        info['gpu_memory'] = "N/A"
    
    return info


def generate_report(
    results: List[Dict],
    latencies: List[float],
    output_path: Path,
    candidates: List[Candidates],
    config: Dict,
    k_values: List[int] = [5, 10]
) -> None:
    """Generate evaluation report.
    
    Args:
        results: List of evaluation results for each query
        latencies: List of latency measurements
        output_path: Path to save the report
        candidates: List of candidate items
        config: Configuration dictionary with parameters, dataset info, etc.
        k_values: List of K values for metrics
    """
    
    # Calculate aggregate metrics
    num_queries = len(results)
    
    # Aggregate metrics across all queries
    aggregate_metrics = {}
    for k in k_values:
        precisions = [r['metrics'][f'precision@{k}'] for r in results]
        recalls = [r['metrics'][f'recall@{k}'] for r in results]
        ndcgs = [r['metrics'][f'ndcg@{k}'] for r in results]
        coverages = [r['metrics'][f'coverage@{k}'] for r in results]
        
        aggregate_metrics[f'precision@{k}'] = {
            'mean': sum(precisions) / num_queries if num_queries > 0 else 0.0,
            'values': precisions,
        }
        aggregate_metrics[f'recall@{k}'] = {
            'mean': sum(recalls) / num_queries if num_queries > 0 else 0.0,
            'values': recalls,
        }
        aggregate_metrics[f'ndcg@{k}'] = {
            'mean': sum(ndcgs) / num_queries if num_queries > 0 else 0.0,
            'values': ndcgs,
        }
        aggregate_metrics[f'coverage@{k}'] = {
            'mean': sum(coverages) / num_queries if num_queries > 0 else 0.0,
            'values': coverages,
        }
    
    mrr_values = [r['metrics']['mrr'] for r in results]
    aggregate_metrics['mrr'] = {
        'mean': sum(mrr_values) / num_queries if num_queries > 0 else 0.0,
        'values': mrr_values,
    }
    
    # Latency statistics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    
    # Get system info
    system_info = get_system_info()
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BM25 Retrieval Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration Section
        f.write("Configuration\n")
        f.write("-" * 80 + "\n\n")
        
        # BM25 Parameters
        f.write("BM25 Parameters:\n")
        f.write(f"  k1: {config.get('bm25_k1', 'N/A')}\n")
        f.write(f"  b:  {config.get('bm25_b', 'N/A')}\n")
        f.write(f"  Cache enabled: {config.get('bm25_cache_enabled', 'N/A')}\n")
        f.write(f"  Cache directory: {config.get('bm25_cache_dir', 'N/A')}\n")
        f.write("\n")
        
        # Retrieval Parameters
        f.write("Retrieval Parameters:\n")
        f.write(f"  Top-K: {config.get('retrieval_top_k', 'N/A')}\n")
        f.write(f"  Evaluation K values: {config.get('k_values', 'N/A')}\n")
        f.write("\n")
        
        # Dataset Configuration
        f.write("Dataset Configuration:\n")
        f.write(f"  Queries file: {config.get('queries_path', 'N/A')}\n")
        f.write(f"  Items file: {config.get('items_path', 'N/A')}\n")
        f.write(f"  Test set file: {config.get('test_query_path', 'N/A')}\n")
        f.write(f"  Number of queries: {config.get('num_queries', 'N/A')}\n")
        f.write(f"  Number of candidates: {config.get('num_candidates', 'N/A')}\n")
        f.write(f"  Number of ground truth pairs: {config.get('num_gt_pairs', 'N/A')}\n")
        f.write("\n")
        
        # Hardware Configuration
        f.write("Hardware Configuration:\n")
        f.write(f"  Platform: {system_info['platform']}\n")
        f.write(f"  Processor: {system_info['processor']}\n")
        f.write(f"  Machine: {system_info['machine']}\n")
        f.write(f"  Architecture: {system_info['architecture']}\n")
        f.write(f"  CPU Count: {system_info['cpu_count']}\n")
        f.write(f"  CPU Frequency: {system_info['cpu_freq']}\n")
        f.write(f"  Total Memory: {system_info['memory_total']}\n")
        f.write(f"  GPU: {system_info['gpu']}\n")
        f.write(f"  GPU Memory: {system_info['gpu_memory']}\n")
        f.write(f"  Python Version: {system_info['python_version']}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Latency statistics
        f.write("Latency Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Latency: {avg_latency*1000:.2f} ms\n")
        f.write(f"Max Latency: {max_latency*1000:.2f} ms\n")
        f.write(f"Min Latency: {min_latency*1000:.2f} ms\n")
        f.write("\n")
        
        # Aggregate metrics
        f.write("Aggregate Metrics:\n")
        f.write("-" * 80 + "\n")
        for k in k_values:
            f.write(f"\nMetrics @ K={k}:\n")
            f.write(f"  Precision@{k}: {aggregate_metrics[f'precision@{k}']['mean']:.4f}\n")
            f.write(f"  Recall@{k}:    {aggregate_metrics[f'recall@{k}']['mean']:.4f}\n")
            f.write(f"  NDCG@{k}:      {aggregate_metrics[f'ndcg@{k}']['mean']:.4f}\n")
            f.write(f"  Coverage@{k}:  {aggregate_metrics[f'coverage@{k}']['mean']:.4f}\n")
        
        f.write(f"\nMRR: {aggregate_metrics['mrr']['mean']:.4f}\n")
        f.write("\n")
        
        # Per-query results
        f.write("Per-Query Results:\n")
        f.write("=" * 80 + "\n")
        
        for result in results:
            query_id = result['query_id']
            query_text = result['query_text']
            metrics = result['metrics']
            
            f.write(f"\nQuery {query_id}: {query_text}\n")
            f.write("-" * 80 + "\n")
            
            # Metrics
            f.write("Metrics:\n")
            for k in k_values:
                f.write(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}, ")
                f.write(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}, ")
                f.write(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}, ")
                f.write(f"Coverage@{k}: {metrics[f'coverage@{k}']:.4f}\n")
            f.write(f"  MRR: {metrics['mrr']:.4f}\n")
            f.write("\n")
            
            # Retrieved results (Top 10)
            f.write("Retrieved Results (Top 10):\n")
            for rank, (item_id, item_name, score) in enumerate(result['retrieved_results'], 1):
                gt_score = result['ground_truth_dict'].get(item_id, 0.0)
                f.write(f"  {rank:2d}. [{item_id}] {item_name[:60]}{'...' if len(item_name) > 60 else ''}\n")
                f.write(f"      BM25 Score: {score:.4f}, GT Score: {gt_score:.1f}\n")
            f.write("\n")
            
            # Ground truth (Top 10)
            f.write("Ground Truth (Top 10):\n")
            candidates_dict_lookup = {c.id: c for c in candidates}
            for rank, (item_id, gt_score) in enumerate(result['ground_truth_items'], 1):
                # Find item name in candidates
                candidate = candidates_dict_lookup.get(item_id)
                item_name = candidate.name if candidate else "N/A"
                
                f.write(f"  {rank:2d}. [{item_id}] {item_name[:60]}{'...' if len(item_name) > 60 else ''}\n")
                f.write(f"      GT Score: {gt_score:.1f}\n")
            f.write("\n")
    
    print(f"\n✓ Evaluation report saved to: {output_path}")


def main():
    """Main evaluation function."""
    # ============================================================================
    # Configuration (modify these parameters as needed)
    # ============================================================================
    
    # Paths
    REPO_ROOT = Path(__file__).resolve().parents[2]
    queries_path = REPO_ROOT / "data" / "test" / "30_queries.csv"
    items_path = REPO_ROOT / "data" / "raw" / "5k_items.csv"
    test_query_path = REPO_ROOT / "data" / "test" / "test_query.csv"
    
    # BM25 parameters
    bm25_k1 = 1.5
    bm25_b = 0.75
    bm25_cache_dir = REPO_ROOT / "data" / "bm25_cache_5k"
    bm25_cache_enabled = True
    
    # Retrieval parameters
    retrieval_top_k = 10  # Top-K results to retrieve for each query
    
    # Evaluation parameters
    k_values = [5, 10]  # K values for metrics (Precision@K, Recall@K, etc.)
    
    # Output path
    output_report_path = REPO_ROOT / "artifacts" / "eval_runs" / "bm25_eval_report.txt"
    
    # ============================================================================
    # Evaluation pipeline
    # ============================================================================
    
    print("=" * 80)
    print("BM25 Retrieval Evaluation")
    print("=" * 80)
    
    # Load queries
    print(f"\n1. Loading queries from: {queries_path}")
    queries = load_queries(queries_path)
    if not queries:
        raise ValueError(f"No valid queries found in {queries_path}")
    print(f"   ✓ Loaded {len(queries)} queries")
    
    # Load ground truth
    print(f"\n2. Loading ground truth from: {test_query_path}")
    ground_truth_all = load_ground_truth(test_query_path)
    print(f"   ✓ Loaded ground truth for {len(ground_truth_all)} queries")
    
    # Load candidates
    print(f"\n3. Loading candidates from: {items_path}")
    candidates, data_source_hash = load_food_candidates(
        items_path,
        cache_dir=bm25_cache_dir,
        cache_enabled=True,
    )
    print(f"   ✓ Loaded {len(candidates)} candidates")
    
    # Initialize BM25 retriever
    print(f"\n4. Initializing BM25 retriever...")
    print(f"   • k1={bm25_k1}, b={bm25_b}")
    print(f"   • Cache dir: {bm25_cache_dir}")
    bm25_retriever = BM25Retriever(
        candidates,
        k1=bm25_k1,
        b=bm25_b,
        cache_dir=bm25_cache_dir,
        cache_enabled=bm25_cache_enabled,
        data_source_hash=data_source_hash,
    )
    print(f"   ✓ BM25 retriever initialized")
    
    # Create candidates lookup for item names
    candidates_dict = {c.id: c for c in candidates}
    
    # Perform search and evaluation
    print(f"\n5. Performing search and evaluation...")
    print(f"   • Retrieval top-K: {retrieval_top_k}")
    print(f"   • Evaluation K values: {k_values}")
    
    results = []
    latencies = []
    
    for i, (query_id, query_text) in enumerate(queries.items(), 1):
        print(f"\n   [{i}/{len(queries)}] Query {query_id}: {query_text[:50]}{'...' if len(query_text) > 50 else ''}")
        
        # Perform search
        search_start = time.time()
        retrieved_results = bm25_retriever.search(query_text, top_k=retrieval_top_k)
        search_time = time.time() - search_start
        latencies.append(search_time)
        
        print(f"      Retrieved {len(retrieved_results)} results in {search_time*1000:.2f}ms")
        
        # Get ground truth for this query
        query_ground_truth = ground_truth_all.get(query_id, {})
        
        # Evaluate
        eval_result = evaluate_query(
            query_id,
            query_text,
            retrieved_results,
            query_ground_truth,
            k_values=k_values,
        )
        results.append(eval_result)
        
        # Print quick metrics
        metrics = eval_result['metrics']
        print(f"      Precision@10: {metrics['precision@10']:.4f}, "
              f"Recall@10: {metrics['recall@10']:.4f}, "
              f"NDCG@10: {metrics['ndcg@10']:.4f}")
    
    # Prepare configuration for report
    config = {
        'bm25_k1': bm25_k1,
        'bm25_b': bm25_b,
        'bm25_cache_enabled': bm25_cache_enabled,
        'bm25_cache_dir': str(bm25_cache_dir),
        'retrieval_top_k': retrieval_top_k,
        'k_values': k_values,
        'queries_path': str(queries_path),
        'items_path': str(items_path),
        'test_query_path': str(test_query_path),
        'num_queries': len(queries),
        'num_candidates': len(candidates),
        'num_gt_pairs': sum(len(gt) for gt in ground_truth_all.values()),
    }
    
    # Generate report
    print(f"\n6. Generating evaluation report...")
    generate_report(results, latencies, output_report_path, candidates, config, k_values=k_values)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    
    print(f"\nLatency Statistics:")
    print(f"  Average: {avg_latency*1000:.2f} ms")
    print(f"  Max:     {max_latency*1000:.2f} ms")
    print(f"  Min:     {min_latency*1000:.2f} ms")
    
    # Calculate aggregate metrics
    for k in k_values:
        precisions = [r['metrics'][f'precision@{k}'] for r in results]
        recalls = [r['metrics'][f'recall@{k}'] for r in results]
        ndcgs = [r['metrics'][f'ndcg@{k}'] for r in results]
        coverages = [r['metrics'][f'coverage@{k}'] for r in results]
        
        print(f"\nMetrics @ K={k}:")
        print(f"  Precision@{k}: {sum(precisions)/len(precisions):.4f}")
        print(f"  Recall@{k}:    {sum(recalls)/len(recalls):.4f}")
        print(f"  NDCG@{k}:      {sum(ndcgs)/len(ndcgs):.4f}")
        print(f"  Coverage@{k}:  {sum(coverages)/len(coverages):.4f}")
    
    mrr_values = [r['metrics']['mrr'] for r in results]
    print(f"\nMRR: {sum(mrr_values)/len(mrr_values):.4f}")
    
    print(f"\n✓ Detailed report saved to: {output_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

