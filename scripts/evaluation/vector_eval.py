#!/usr/bin/env python3
"""Vector retrieval evaluation script.

This script:
1. Loads queries from data/test/30_queries.csv
2. Performs Vector search on data/raw/5k_items.csv (using cached embeddings/index)
3. Evaluates results against LLM-scored test set data/test/test_query.csv
4. Generates comprehensive evaluation report

Evaluation metrics:
- Precision@K, Recall@K, NDCG@K, MRR
- Average/Max/Min latency
- Per-query results and ground truth comparison
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
from src.vector_retrieval import VectorRetriever
from scripts.evaluation.evaluation_utils import (
    load_queries,
    load_ground_truth,
    evaluate_query,
    generate_evaluation_report,
)


def main():
    """Main evaluation function."""
    # ============================================================================
    # Configuration (modify these parameters as needed)
    # ============================================================================
    
    # Paths
    queries_path = REPO_ROOT / "data" / "test" / "30_queries.csv"
    items_path = REPO_ROOT / "data" / "raw" / "5k_items.csv"
    test_query_path = REPO_ROOT / "data" / "test" / "test_query.csv"
    
    # Vector retrieval parameters
    # Choose one: API mode or Local model mode
    
    # Option 1: Local model mode (recommended for evaluation)
    use_local_model = True
    local_model_name = "BAAI/bge-m3"  # or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    normalize_embeddings = True
    
    # Option 2: API mode (if you want to use OpenAI API)
    # Uncomment and configure these if using API mode:
    # use_local_model = False
    # vector_api_base = "https://api.openai.com/v1"
    # vector_api_key = "your-api-key"
    # vector_model_name = "text-embedding-3-small"
    # vector_dimensions = 1536
    # vector_max_tokens_per_request = 8192
    # vector_max_items_per_batch = None
    # vector_rpm_limit = 300
    # vector_timeout = 120.0
    
    # HNSW index parameters
    vector_hnsw_index_path = REPO_ROOT / "data" / "vector_indices_5k"  # Directory or file path
    vector_use_hnsw = True
    vector_hnsw_m = 32
    vector_hnsw_ef_construction = 100
    vector_hnsw_ef_search = 64
    
    # Embeddings caching parameters
    vector_embeddings_dir = REPO_ROOT / "data" / "vector_indices_5k"
    vector_cache_embeddings = True
    
    # Query embedding parameters (optional, defaults to local_model_name)
    vector_query_embedding_model = None  # None = use local_model_name
    vector_query_embedding_device = None  # None = auto-detect
    
    # Retrieval parameters
    retrieval_top_k = 10  # Top-K results to retrieve for each query
    
    # Evaluation parameters
    k_values = [5, 10]  # K values for metrics (Precision@K, Recall@K, etc.)
    
    # Output path
    output_report_path = REPO_ROOT / "artifacts" / "eval_runs" / "vector_eval_report_BAAI_bge_m3.txt"
    
    # ============================================================================
    # Evaluation pipeline
    # ============================================================================
    
    print("=" * 80)
    print("Vector Retrieval Evaluation")
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
    candidates, _ = load_food_candidates(
        items_path,
        cache_dir=None,  # Not using BM25 cache for vector eval
        cache_enabled=False,
    )
    print(f"   ✓ Loaded {len(candidates)} candidates")
    
    # Initialize Vector retriever
    print(f"\n4. Initializing Vector retriever...")
    
    if use_local_model:
        print(f"   • Mode: Local model")
        print(f"   • Model: {local_model_name}")
        print(f"   • Normalize embeddings: {normalize_embeddings}")
        print(f"   • HNSW index path: {vector_hnsw_index_path}")
        
        vector_retriever = VectorRetriever(
            candidates,
            # Local model parameters
            local_model_name=local_model_name,
            normalize_embeddings=normalize_embeddings,
            # HNSW indexing parameters
            use_hnsw=vector_use_hnsw,
            index_path=vector_hnsw_index_path,
            hnsw_m=vector_hnsw_m,
            hnsw_ef_construction=vector_hnsw_ef_construction,
            hnsw_ef_search=vector_hnsw_ef_search,
            # Embeddings caching parameters
            embeddings_dir=vector_embeddings_dir,
            cache_embeddings=vector_cache_embeddings,
            # Query embedding parameters
            query_embedding_model=vector_query_embedding_model,
            query_embedding_device=vector_query_embedding_device,
        )
    else:
        # API mode - make sure to uncomment and set API parameters above
        raise ValueError(
            "API mode is not configured. Please uncomment and set API parameters in the config section, "
            "or set use_local_model = True to use local model mode."
        )
    
    print(f"   ✓ Vector retriever initialized")
    
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
        retrieved_results = vector_retriever.search(query_text, top_k=retrieval_top_k)
        search_time = time.time() - search_start
        latencies.append(search_time)
        
        print(f"      Retrieved {len(retrieved_results)} results in {search_time*1000:.2f}ms")
        
        # Get ground truth for this query
        query_ground_truth = ground_truth_all.get(query_id, {})
        
        # Prepare data for evaluation
        retrieved_items = [candidate.id for candidate, _ in retrieved_results]
        retrieved_scores = [score for _, score in retrieved_results]
        retrieved_results_formatted = [
            (candidate.id, candidate.name, score)
            for candidate, score in retrieved_results
        ]
        
        # Evaluate
        eval_result = evaluate_query(
            query_id,
            query_text,
            retrieved_items,
            retrieved_scores,
            retrieved_results_formatted,
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
    if use_local_model:
        method_params = {
            'mode': 'Local model',
            'model_name': local_model_name,
            'normalize_embeddings': normalize_embeddings,
            'use_hnsw': vector_use_hnsw,
            'hnsw_m': vector_hnsw_m,
            'hnsw_ef_construction': vector_hnsw_ef_construction,
            'hnsw_ef_search': vector_hnsw_ef_search,
            'cache_embeddings': vector_cache_embeddings,
            'index_path': str(vector_hnsw_index_path),
            'embeddings_dir': str(vector_embeddings_dir),
        }
        if vector_query_embedding_model:
            method_params['query_embedding_model'] = vector_query_embedding_model
    else:
        # API mode (not currently used, but kept for future use)
        method_params = {
            'mode': 'API',
            'use_hnsw': vector_use_hnsw,
            'hnsw_m': vector_hnsw_m,
            'hnsw_ef_construction': vector_hnsw_ef_construction,
            'hnsw_ef_search': vector_hnsw_ef_search,
            'cache_embeddings': vector_cache_embeddings,
            'index_path': str(vector_hnsw_index_path),
            'embeddings_dir': str(vector_embeddings_dir),
        }
    
    config = {
        'method_parameters': method_params,
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
    generate_evaluation_report(
        results,
        latencies,
        output_report_path,
        candidates,
        config,
        method_name="Vector",
        k_values=k_values,
    )
    
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
        
        print(f"\nMetrics @ K={k}:")
        print(f"  Precision@{k}: {sum(precisions)/len(precisions):.4f}")
        print(f"  Recall@{k}:    {sum(recalls)/len(recalls):.4f}")
        print(f"  NDCG@{k}:      {sum(ndcgs)/len(ndcgs):.4f}")
    
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

