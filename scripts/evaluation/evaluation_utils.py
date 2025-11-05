#!/usr/bin/env python3
"""Common evaluation utilities for retrieval evaluation.

This module provides reusable functions for:
- Evaluation metrics (Precision@K, Recall@K, NDCG@K, MRR, Coverage@K)
- Report generation
- System information collection
- Data loading utilities
"""

from __future__ import annotations

import csv
import math
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from src.bm25_retrieval import Candidates


# ============================================================================
# Data Loading Utilities
# ============================================================================

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


# ============================================================================
# Evaluation Metrics
# ============================================================================

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
    retrieved_items: List[str],
    retrieved_scores: List[float],
    retrieved_results: List[Tuple[str, str, float]],  # (item_id, item_name, score)
    ground_truth: Dict[str, float],
    k_values: List[int] = [5, 10]
) -> Dict:
    """Evaluate a single query.
    
    Args:
        query_id: Query ID
        query_text: Query text
        retrieved_items: List of retrieved item IDs
        retrieved_scores: List of retrieval scores
        retrieved_results: List of (item_id, item_name, score) tuples for top results
        ground_truth: Dictionary mapping item_id to relevance score
        k_values: List of K values for metrics
        
    Returns:
        Dictionary with evaluation metrics and results
    """
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
        'retrieved_results': retrieved_results[:10],  # Top 10
        'ground_truth_items': top_gt_items,
        'ground_truth_dict': ground_truth,  # Full ground truth dict for score lookup
        'metrics': metrics,
    }


# ============================================================================
# System Information
# ============================================================================

def get_system_info() -> Dict[str, str]:
    """Get system hardware and software information."""
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


# ============================================================================
# Report Generation
# ============================================================================

def generate_evaluation_report(
    results: List[Dict],
    latencies: List[float],
    output_path: Path,
    candidates: List[Candidates],
    config: Dict,
    method_name: str = "Retrieval",
    k_values: List[int] = [5, 10]
) -> None:
    """Generate comprehensive evaluation report.
    
    Args:
        results: List of evaluation results for each query
        latencies: List of latency measurements
        output_path: Path to save the report
        candidates: List of candidate items
        config: Configuration dictionary with parameters, dataset info, etc.
        method_name: Name of the retrieval method (e.g., "BM25", "Vector")
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
        f.write(f"{method_name} Retrieval Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration Section
        f.write("Configuration\n")
        f.write("-" * 80 + "\n\n")
        
        # Method-specific parameters
        method_params = config.get('method_parameters', {})
        if method_params:
            f.write(f"{method_name} Parameters:\n")
            for key, value in method_params.items():
                f.write(f"  {key}: {value}\n")
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
        
        candidates_dict = {c.id: c for c in candidates}
        
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
                f.write(f"      Score: {score:.4f}, GT Score: {gt_score:.1f}\n")
            f.write("\n")
            
            # Ground truth (Top 10)
            f.write("Ground Truth (Top 10):\n")
            for rank, (item_id, gt_score) in enumerate(result['ground_truth_items'], 1):
                # Find item name in candidates
                candidate = candidates_dict.get(item_id)
                item_name = candidate.name if candidate else "N/A"
                
                f.write(f"  {rank:2d}. [{item_id}] {item_name[:60]}{'...' if len(item_name) > 60 else ''}\n")
                f.write(f"      GT Score: {gt_score:.1f}\n")
            f.write("\n")
    
    print(f"\n✓ Evaluation report saved to: {output_path}")

