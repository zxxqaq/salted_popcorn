#!/usr/bin/env python3
"""Hybrid retrieval evaluation script.

This script:
1. Loads queries from data/test/30_queries.csv
2. Performs hybrid search (BM25 + Vector + RRF + Cross-Encoder) on data/raw/5k_items.csv
3. Evaluates results against LLM-scored test set data/test/test_query.csv
4. Generates comprehensive evaluation report with detailed timing breakdown

Evaluation metrics:
- Precision@K, Recall@K, NDCG@K, MRR
- Average/Max/Min latency (total and per-stage)
- Per-query results and ground truth comparison
- Detailed timing breakdown for each module (BM25, Vector, RRF, Reranker)
"""

from __future__ import annotations

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

from src.bm25_retrieval import Candidates, load_food_candidates
from src.hybrid_retrieval_no_llm import HybridRetriever, HybridRetrievalResult
from scripts.evaluation.evaluation_utils import (
    load_queries,
    load_ground_truth,
    evaluate_query,
    get_system_info,
    RELEVANCE_THRESHOLD,
    NDCG_THRESHOLD,
)


def evaluate_hybrid_query(
    query_id: str,
    query_text: str,
    result: HybridRetrievalResult,
    ground_truth: Dict[str, float],
    candidates: List[Candidates],
    k_values: List[int] = [5, 10]
) -> Dict:
    """Evaluate a single hybrid retrieval query.
    
    Args:
        query_id: Query ID
        query_text: Query text
        result: HybridRetrievalResult from hybrid search
        ground_truth: Dictionary mapping item_id to relevance score
        candidates: List of all candidates
        k_values: List of K values for metrics
        
    Returns:
        Dictionary with evaluation metrics and results
    """
    # Extract top-10 results (item_id, item_name, reranker_score)
    # result.top_10 is List[Tuple[Candidates, float]]
    retrieved_items = [item.id for item, _ in result.top_10]
    retrieved_scores = [score for _, score in result.top_10]
    retrieved_results = [
        (item.id, item.name, score)
        for item, score in result.top_10
    ]
    
    # Use standard evaluate_query function
    eval_result = evaluate_query(
        query_id=query_id,
        query_text=query_text,
        retrieved_items=retrieved_items,
        retrieved_scores=retrieved_scores,
        retrieved_results=retrieved_results,
        ground_truth=ground_truth,
        k_values=k_values
    )
    
    # Add hybrid-specific timing information
    eval_result['timing'] = {
        'bm25_time': result.bm25_time,
        'vector_time': result.vector_time,
        'rrf_time': result.rrf_time,
        'rerank_time': result.rerank_time,
        'total_time': result.total_time,
        'ce_timing_info': result.ce_timing_info,
    }
    
    return eval_result


def generate_hybrid_evaluation_report(
    results: List[Dict],
    latencies: List[float],
    output_path: Path,
    candidates: List[Candidates],
    config: Dict,
    k_values: List[int] = [5, 10]
) -> None:
    """Generate comprehensive hybrid evaluation report with detailed timing.
    
    Args:
        results: List of evaluation results for each query
        latencies: List of total latency measurements
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
    
    mrr_values = [r['metrics']['mrr'] for r in results]
    aggregate_metrics['mrr'] = {
        'mean': sum(mrr_values) / num_queries if num_queries > 0 else 0.0,
        'values': mrr_values,
    }
    
    # Latency statistics (total)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    
    # Per-stage timing statistics
    bm25_times = [r['timing']['bm25_time'] for r in results]
    vector_times = [r['timing']['vector_time'] for r in results]
    rrf_times = [r['timing']['rrf_time'] for r in results]
    rerank_times = [r['timing']['rerank_time'] for r in results]
    
    avg_bm25_time = sum(bm25_times) / len(bm25_times) if bm25_times else 0.0
    avg_vector_time = sum(vector_times) / len(vector_times) if vector_times else 0.0
    avg_rrf_time = sum(rrf_times) / len(rrf_times) if rrf_times else 0.0
    avg_rerank_time = sum(rerank_times) / len(rerank_times) if rerank_times else 0.0
    
    max_bm25_time = max(bm25_times) if bm25_times else 0.0
    max_vector_time = max(vector_times) if vector_times else 0.0
    max_rrf_time = max(rrf_times) if rrf_times else 0.0
    max_rerank_time = max(rerank_times) if rerank_times else 0.0
    
    min_bm25_time = min(bm25_times) if bm25_times else 0.0
    min_vector_time = min(vector_times) if vector_times else 0.0
    min_rrf_time = min(rrf_times) if rrf_times else 0.0
    min_rerank_time = min(rerank_times) if rerank_times else 0.0
    
    # Cross-Encoder detailed timing (if available)
    ce_timing_available = False
    ce_batch_times = []
    ce_tokenization_times = []
    ce_scoring_times = []
    
    for r in results:
        ce_info = r['timing'].get('ce_timing_info')
        if ce_info:
            ce_timing_available = True
            if 'batch_times' in ce_info:
                ce_batch_times.extend(ce_info['batch_times'])
            if 'tokenization_time' in ce_info:
                ce_tokenization_times.append(ce_info['tokenization_time'])
            if 'scoring_time' in ce_info:
                ce_scoring_times.append(ce_info['scoring_time'])
    
    # Get system info
    system_info = get_system_info()
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Hybrid Retrieval Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration Section
        f.write("Configuration\n")
        f.write("-" * 80 + "\n\n")
        
        # Hybrid method parameters
        method_params = config.get('method_parameters', {})
        if method_params:
            f.write("Hybrid Retrieval Parameters:\n")
            for key, value in method_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Retrieval Parameters
        f.write("Retrieval Parameters:\n")
        f.write(f"  Retrieval Top-K (per method): {config.get('retrieval_top_k', 'N/A')}\n")
        f.write(f"  Final Top-K 1: {config.get('final_top_k_1', 'N/A')}\n")
        f.write(f"  Final Top-K 2: {config.get('final_top_k_2', 'N/A')}\n")
        f.write(f"  Evaluation K values: {config.get('k_values', 'N/A')}\n")
        f.write(f"  Relevance Threshold: {RELEVANCE_THRESHOLD} (items with score >= {RELEVANCE_THRESHOLD} are considered relevant)\n")
        f.write(f"  NDCG Threshold: {NDCG_THRESHOLD} (items with score >= {NDCG_THRESHOLD} are used for IDCG calculation)\n")
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
        
        # Latency statistics (total)
        f.write("Total Latency Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Latency: {avg_latency*1000:.2f} ms\n")
        f.write(f"Max Latency: {max_latency*1000:.2f} ms\n")
        f.write(f"Min Latency: {min_latency*1000:.2f} ms\n")
        f.write("\n")
        
        # Per-stage timing statistics
        f.write("Per-Stage Timing Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write("BM25 Retrieval:\n")
        f.write(f"  Average: {avg_bm25_time*1000:.2f} ms\n")
        f.write(f"  Max: {max_bm25_time*1000:.2f} ms\n")
        f.write(f"  Min: {min_bm25_time*1000:.2f} ms\n")
        f.write("\n")
        
        f.write("Vector Retrieval:\n")
        f.write(f"  Average: {avg_vector_time*1000:.2f} ms\n")
        f.write(f"  Max: {max_vector_time*1000:.2f} ms\n")
        f.write(f"  Min: {min_vector_time*1000:.2f} ms\n")
        f.write("\n")
        
        f.write("RRF Fusion / Merge:\n")
        f.write(f"  Average: {avg_rrf_time*1000:.2f} ms\n")
        f.write(f"  Max: {max_rrf_time*1000:.2f} ms\n")
        f.write(f"  Min: {min_rrf_time*1000:.2f} ms\n")
        f.write("\n")
        
        f.write("Cross-Encoder Re-ranking:\n")
        f.write(f"  Average: {avg_rerank_time*1000:.2f} ms\n")
        f.write(f"  Max: {max_rerank_time*1000:.2f} ms\n")
        f.write(f"  Min: {min_rerank_time*1000:.2f} ms\n")
        f.write("\n")
        
        # Cross-Encoder detailed timing (if available)
        if ce_timing_available:
            f.write("Cross-Encoder Detailed Timing:\n")
            if ce_tokenization_times:
                avg_tokenization = sum(ce_tokenization_times) / len(ce_tokenization_times)
                f.write(f"  Tokenization (avg): {avg_tokenization*1000:.2f} ms\n")
            if ce_scoring_times:
                avg_scoring = sum(ce_scoring_times) / len(ce_scoring_times)
                f.write(f"  Scoring (avg): {avg_scoring*1000:.2f} ms\n")
            if ce_batch_times:
                avg_batch = sum(ce_batch_times) / len(ce_batch_times)
                max_batch = max(ce_batch_times)
                min_batch = min(ce_batch_times)
                f.write(f"  Batch Processing (avg): {avg_batch*1000:.2f} ms\n")
                f.write(f"  Batch Processing (max): {max_batch*1000:.2f} ms\n")
                f.write(f"  Batch Processing (min): {min_batch*1000:.2f} ms\n")
                f.write(f"  Total batches processed: {len(ce_batch_times)}\n")
            f.write("\n")
        
        # Aggregate metrics
        f.write("Aggregate Metrics:\n")
        f.write("-" * 80 + "\n")
        for k in k_values:
            f.write(f"\nMetrics @ K={k}:\n")
            f.write(f"  Precision@{k}: {aggregate_metrics[f'precision@{k}']['mean']:.4f}\n")
            f.write(f"  Recall@{k}:    {aggregate_metrics[f'recall@{k}']['mean']:.4f}\n")
            f.write(f"  NDCG@{k}:      {aggregate_metrics[f'ndcg@{k}']['mean']:.4f}\n")
        
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
            timing = result['timing']
            
            f.write(f"\nQuery {query_id}: {query_text}\n")
            f.write("-" * 80 + "\n")
            
            # Metrics
            f.write("Metrics:\n")
            for k in k_values:
                f.write(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}, ")
                f.write(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}, ")
                f.write(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}\n")
            f.write(f"  MRR: {metrics['mrr']:.4f}\n")
            f.write("\n")
            
            # Timing breakdown
            f.write("Timing Breakdown:\n")
            f.write(f"  BM25: {timing['bm25_time']*1000:.2f} ms\n")
            f.write(f"  Vector: {timing['vector_time']*1000:.2f} ms\n")
            f.write(f"  RRF/Merge: {timing['rrf_time']*1000:.2f} ms\n")
            f.write(f"  Rerank: {timing['rerank_time']*1000:.2f} ms\n")
            f.write(f"  Total: {timing['total_time']*1000:.2f} ms\n")
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


def main():
    """Main evaluation function."""
    # ============================================================================
    # Configuration (modify these parameters as needed)
    # ============================================================================
    
    # Paths
    queries_path = REPO_ROOT / "data" / "test" / "30_queries.csv"
    items_path = REPO_ROOT / "data" / "raw" / "5k_items.csv"
    test_query_path = REPO_ROOT / "data" / "test" / "test_query.csv"
    
    # BM25 parameters
    bm25_k1 = 1.5
    bm25_b = 0.75
    bm25_cache_dir = REPO_ROOT / "data" / "bm25_cache_5k"
    bm25_cache_enabled = True
    
    # Vector retrieval parameters (local model)
    local_model_name = "BAAI/bge-m3"  # or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    normalize_embeddings = True
    
    # HNSW index parameters
    vector_hnsw_index_path = REPO_ROOT / "data" / "vector_indices_5k" / "hnsw_BAAI_bge_m3_dim1024_5000.index"  # Full path to .index file
    vector_use_hnsw = True
    vector_hnsw_m = 32
    vector_hnsw_ef_construction = 100
    vector_hnsw_ef_search = 64
    
    # Embeddings caching parameters
    vector_embeddings_dir = REPO_ROOT / "data" / "vector_indices_5k"  # Directory for embeddings cache
    vector_cache_embeddings = True
    
    # Query embedding parameters (optional, defaults to local_model_name if not set)
    vector_query_embedding_model = None  # Defaults to local_model_name if use_local_model is True
    vector_query_embedding_device = None  # "cpu", "cuda", "mps", or None for auto
    
    # Retrieval parameters
    retrieval_top_k = 50  # Top-K for each retrieval method (BM25 and Vector)
    
    # RRF fusion parameters
    use_rrf = True
    rrf_k = 60
    rrf_top_k = 20  # Top-K results after RRF fusion (input to Cross-Encoder)
    
    # Final output parameters
    final_top_k_1 = 5  # First top-K result set (e.g., top-5)
    final_top_k_2 = 10  # Second top-K result set (e.g., top-10)
    
    # Cross-Encoder re-ranking parameters
    reranker_model = "BAAI/bge-reranker-v2-m3"
    reranker_device = "mps"  # "mps", "cuda", "cpu", or None for auto
    reranker_batch_size = 64  # 32 or 64 for Mac MPS
    reranker_top_k = 10  # None = return all scored items
    reranker_tokenization_cache_dir = REPO_ROOT / "data" / "reranker_tokenization_cache_5k"
    reranker_tokenization_cache_enabled = True
    reranker_max_concurrent_batches = 1
    
    # Pre-filtering parameters (before Cross-Encoder re-ranking)
    reranker_prefilter_enabled = True
    reranker_prefilter_min_score = 0.016393
    reranker_prefilter_score_diff_threshold = 0.000598
    reranker_prefilter_min_items = 20
    
    # Evaluation parameters
    k_values = [5, 10]  # K values for metrics
    
    # Output path
    output_report_path = REPO_ROOT / "artifacts" / "eval_runs" / "hybrid_eval_report.txt"
    
    # ============================================================================
    # Evaluation pipeline
    # ============================================================================
    
    print("=" * 80)
    print("Hybrid Retrieval Evaluation")
    print("=" * 80)
    
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. Loading queries from: {queries_path}")
    queries = load_queries(queries_path)
    if not queries:
        raise ValueError(f"No valid queries found in {queries_path}")
    print(f"   ✓ Loaded {len(queries)} queries")
    
    print(f"\n2. Loading ground truth from: {test_query_path}")
    ground_truth_all = load_ground_truth(test_query_path)
    print(f"   ✓ Loaded ground truth for {len(ground_truth_all)} queries")
    
    print(f"\n3. Loading candidates from: {items_path}")
    candidates, data_source_hash = load_food_candidates(
        items_path,
        cache_dir=bm25_cache_dir,
        cache_enabled=bm25_cache_enabled,
    )
    print(f"   ✓ Loaded {len(candidates)} candidates")
    
    print(f"\n4. Initializing Hybrid retriever...")
    print(f"   • Vector mode: Local model")
    print(f"   • Vector model: {local_model_name}")
    print(f"   • Normalize embeddings: {normalize_embeddings}")
    print(f"   • HNSW index path: {vector_hnsw_index_path}")
    
    hybrid_retriever = HybridRetriever(
        candidates,
        # BM25 parameters
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        bm25_cache_dir=bm25_cache_dir,
        bm25_cache_enabled=bm25_cache_enabled,
        bm25_data_source_hash=data_source_hash,
        # Vector retrieval parameters (local model)
        vector_local_model_name=local_model_name,
        vector_normalize_embeddings=normalize_embeddings,
        vector_hnsw_index_path=vector_hnsw_index_path,
        vector_use_hnsw=vector_use_hnsw,
        vector_hnsw_m=vector_hnsw_m,
        vector_hnsw_ef_construction=vector_hnsw_ef_construction,
        vector_hnsw_ef_search=vector_hnsw_ef_search,
        vector_embeddings_dir=vector_embeddings_dir,
        vector_cache_embeddings=vector_cache_embeddings,
        vector_query_embedding_model=vector_query_embedding_model,
        # Retrieval parameters
        retrieval_top_k=retrieval_top_k,
        # RRF fusion parameters
        use_rrf=use_rrf,
        rrf_k=rrf_k,
        rrf_top_k=rrf_top_k,
        # Final output parameters
        final_top_k_1=final_top_k_1,
        final_top_k_2=final_top_k_2,
        # Cross-Encoder re-ranking parameters
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        reranker_batch_size=reranker_batch_size,
        reranker_top_k=reranker_top_k,
        reranker_tokenization_cache_dir=reranker_tokenization_cache_dir,
        reranker_tokenization_cache_enabled=reranker_tokenization_cache_enabled,
        reranker_max_concurrent_batches=reranker_max_concurrent_batches,
        # Pre-filtering parameters
        reranker_prefilter_enabled=reranker_prefilter_enabled,
        reranker_prefilter_min_score=reranker_prefilter_min_score,
        reranker_prefilter_score_diff_threshold=reranker_prefilter_score_diff_threshold,
        reranker_prefilter_min_items=reranker_prefilter_min_items,
    )
    
    print(f"   ✓ Hybrid retriever initialized")
    
    print(f"\n5. Performing search and evaluation...")
    results = []
    latencies = []
    
    for i, (query_id, query_text) in enumerate(queries.items(), 1):
        print(f"\n[{i}/{len(queries)}] Query {query_id}: {query_text[:60]}{'...' if len(query_text) > 60 else ''}")
        
        search_start_time = time.time()
        hybrid_result = hybrid_retriever.search(query_text, query_id=query_id)
        latency = time.time() - search_start_time
        latencies.append(latency)
        
        ground_truth_for_query = ground_truth_all.get(query_id, {})
        
        eval_result = evaluate_hybrid_query(
            query_id,
            query_text,
            hybrid_result,
            ground_truth_for_query,
            candidates,
            k_values=k_values
        )
        results.append(eval_result)
        
        metrics = eval_result['metrics']
        timing = eval_result['timing']
        print(f"      Precision@10: {metrics['precision@10']:.4f}, "
              f"Recall@10: {metrics['recall@10']:.4f}, "
              f"NDCG@10: {metrics['ndcg@10']:.4f}")
        print(f"      Timing: BM25={timing['bm25_time']*1000:.1f}ms, "
              f"Vector={timing['vector_time']*1000:.1f}ms, "
              f"RRF={timing['rrf_time']*1000:.1f}ms, "
              f"Rerank={timing['rerank_time']*1000:.1f}ms, "
              f"Total={timing['total_time']*1000:.1f}ms")
    
    # Prepare configuration for report
    method_params = {
        'mode': 'Hybrid (BM25 + Vector + RRF + Cross-Encoder)',
        'vector_mode': 'Local Model',
        'vector_local_model_name': local_model_name,
        'vector_normalize_embeddings': normalize_embeddings,
        'vector_use_hnsw': vector_use_hnsw,
        'vector_hnsw_m': vector_hnsw_m,
        'vector_hnsw_ef_construction': vector_hnsw_ef_construction,
        'vector_hnsw_ef_search': vector_hnsw_ef_search,
        'vector_cache_embeddings': vector_cache_embeddings,
        'vector_hnsw_index_path': str(vector_hnsw_index_path),
        'vector_embeddings_dir': str(vector_embeddings_dir),
    }
    if vector_query_embedding_model:
        method_params['vector_query_embedding_model'] = vector_query_embedding_model
    if vector_query_embedding_device:
        method_params['vector_query_embedding_device'] = vector_query_embedding_device
    
    method_params.update({
        'bm25_k1': bm25_k1,
        'bm25_b': bm25_b,
        'bm25_cache_enabled': bm25_cache_enabled,
        'bm25_cache_dir': str(bm25_cache_dir),
        'use_rrf': use_rrf,
        'rrf_k': rrf_k,
        'rrf_top_k': rrf_top_k,
        'reranker_model': reranker_model,
        'reranker_device': reranker_device or 'auto',
        'reranker_batch_size': reranker_batch_size,
        'reranker_max_concurrent_batches': reranker_max_concurrent_batches,
        'reranker_prefilter_enabled': reranker_prefilter_enabled,
        'reranker_prefilter_min_items': reranker_prefilter_min_items,
    })
    
    config = {
        'method': 'Hybrid Retrieval',
        'method_parameters': method_params,
        'retrieval_top_k': retrieval_top_k,
        'final_top_k_1': final_top_k_1,
        'final_top_k_2': final_top_k_2,
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
    generate_hybrid_evaluation_report(results, latencies, output_report_path, candidates, config, k_values=k_values)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"Average Total Latency: {avg_latency*1000:.2f} ms")
    
    # Per-stage timing summary
    bm25_times = [r['timing']['bm25_time'] for r in results]
    vector_times = [r['timing']['vector_time'] for r in results]
    rrf_times = [r['timing']['rrf_time'] for r in results]
    rerank_times = [r['timing']['rerank_time'] for r in results]
    
    print(f"\nPer-Stage Timing (Average):")
    print(f"  BM25: {sum(bm25_times)/len(bm25_times)*1000:.2f} ms")
    print(f"  Vector: {sum(vector_times)/len(vector_times)*1000:.2f} ms")
    print(f"  RRF/Merge: {sum(rrf_times)/len(rrf_times)*1000:.2f} ms")
    print(f"  Rerank: {sum(rerank_times)/len(rerank_times)*1000:.2f} ms")
    
    for k in k_values:
        mean_precision = sum(r['metrics'][f'precision@{k}'] for r in results) / len(results)
        mean_recall = sum(r['metrics'][f'recall@{k}'] for r in results) / len(results)
        mean_ndcg = sum(r['metrics'][f'ndcg@{k}'] for r in results) / len(results)
        print(f"\nAverage Metrics @ K={k}:")
        print(f"  Precision@{k}: {mean_precision:.4f}, Recall@{k}: {mean_recall:.4f}, NDCG@{k}: {mean_ndcg:.4f}")
    
    print("\n✓ Hybrid evaluation completed.")


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

