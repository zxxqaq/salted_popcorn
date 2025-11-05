#!/usr/bin/env python3
"""
Analyze score distribution to provide recommendations for pre-filter thresholds.

This script:
1. Runs hybrid retrieval on sample queries (with pre-filter disabled)
2. Collects BM25, Vector, RRF/max scores from retrieval results
3. Analyzes score distributions (min, max, mean, median, percentiles)
4. Analyzes adjacent score differences
5. Provides recommended threshold values based on actual data

Usage:
    python scripts/analyze_score_distribution.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import statistics
from typing import List, Tuple, Dict, Optional

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

import logging
import csv

logging.basicConfig(level=logging.INFO)

from src.hybrid_retrieval_no_llm import HybridRetriever, load_food_candidates_for_hybrid


def analyze_scores(results: List[Tuple], score_type: str) -> Optional[Dict]:
    """
    Analyze score distribution statistics.
    
    Args:
        results: List of (item, score) tuples
        score_type: Type of score (e.g., "BM25", "Vector", "RRF")
        
    Returns:
        Dictionary with statistics (min, max, mean, median, percentiles) or None if empty
    """
    scores = [score for _, score in results]
    if not scores:
        return None

    try:
        quantiles_p4 = statistics.quantiles(scores, n=4)
        quantiles_p10 = statistics.quantiles(scores, n=10)
        quantiles_p20 = statistics.quantiles(scores, n=20)
    except Exception:
        # If quantiles calculation fails, use fallback values
        quantiles_p4 = [scores[0], scores[0], scores[-1]]
        quantiles_p10 = [scores[0]] * 9
        quantiles_p20 = [scores[0]] * 19

    return {
        'type': score_type,
        'count': len(scores),
        'min': min(scores),
        'max': max(scores),
        'mean': statistics.mean(scores),
        'median': statistics.median(scores),
        'p25': quantiles_p4[0] if len(quantiles_p4) > 0 else scores[0],
        'p75': quantiles_p4[2] if len(quantiles_p4) > 2 else scores[-1],
        'p90': quantiles_p10[8] if len(quantiles_p10) > 8 else scores[-1],
        'p95': quantiles_p20[18] if len(quantiles_p20) > 18 else scores[-1],
    }


def analyze_score_differences(results: List[Tuple]) -> Optional[Dict]:
    """
    Analyze adjacent score differences.
    
    Args:
        results: List of (item, score) tuples (should be sorted by score descending)
        
    Returns:
        Dictionary with difference statistics or None if insufficient data
    """
    scores = [score for _, score in results]
    if len(scores) < 2:
        return None

    diffs = []
    for i in range(len(scores) - 1):
        diff = scores[i] - scores[i + 1]
        diffs.append(diff)

    if not diffs:
        return None

    try:
        quantiles_p10 = statistics.quantiles(diffs, n=10)
        quantiles_p20 = statistics.quantiles(diffs, n=20)
    except Exception:
        quantiles_p10 = [diffs[0]] * 9
        quantiles_p20 = [diffs[0]] * 19

    median_diff = statistics.median(diffs)
    return {
        'count': len(diffs),
        'min': min(diffs),
        'max': max(diffs),
        'mean': statistics.mean(diffs),
        'median': median_diff,
        'p90': quantiles_p10[8] if len(quantiles_p10) > 8 else diffs[-1],
        'p95': quantiles_p20[18] if len(quantiles_p20) > 18 else diffs[-1],
        'large_drops': sum(1 for d in diffs if d > median_diff * 3),
    }


def calculate_combined_score(
    item_id: str,
    bm25_results: List[Tuple],
    vector_results: List[Tuple],
    use_rrf: bool,
    rrf_k: int,
) -> float:
    """
    Calculate combined score (RRF or max) for an item.
    
    Args:
        item_id: Item ID
        bm25_results: BM25 retrieval results with scores
        vector_results: Vector retrieval results with scores
        use_rrf: Whether to use RRF fusion
        rrf_k: RRF k parameter
        
    Returns:
        Combined score
    """
    bm25_score_map = {item.id: score for item, score in bm25_results}
    vector_score_map = {item.id: score for item, score in vector_results}
    
    if use_rrf:
        # Calculate RRF score
        rrf_score = 0.0
        if item_id in bm25_score_map:
            bm25_rank = next(
                (i for i, (c, _) in enumerate(bm25_results) if c.id == item_id),
                len(bm25_results)
            )
            rrf_score += 1.0 / (rrf_k + bm25_rank + 1)
        if item_id in vector_score_map:
            vector_rank = next(
                (i for i, (c, _) in enumerate(vector_results) if c.id == item_id),
                len(vector_results)
            )
            rrf_score += 1.0 / (rrf_k + vector_rank + 1)
        return rrf_score
    else:
        # Use max of BM25 and Vector scores
        bm25_score = bm25_score_map.get(item_id, 0.0)
        vector_score = vector_score_map.get(item_id, 0.0)
        return max(bm25_score, vector_score)


def load_retriever_config() -> Dict:
    """Load retriever configuration from environment variables."""
    # BM25 parameters
    bm25_k1_str = os.getenv("BM25_K1", "1.5").strip()
    bm25_k1 = float(bm25_k1_str) if bm25_k1_str else 1.5
    
    bm25_b_str = os.getenv("BM25_B", "0.75").strip()
    bm25_b = float(bm25_b_str) if bm25_b_str else 0.75
    
    # Vector parameters
    vector_local_model_name = os.getenv("VECTOR_LOCAL_MODEL_NAME", "").strip() or None
    vector_api_base = os.getenv("VECTOR_API_BASE", "").strip() or None
    vector_api_key = os.getenv("VECTOR_API_KEY") or os.getenv("OPENAI_API_KEY")
    vector_api_key = vector_api_key.strip() if vector_api_key else None
    vector_model_name = os.getenv("VECTOR_MODEL_NAME", "text-embedding-3-small").strip()
    
    vector_dimensions_str = os.getenv("OPENAI_EMBEDDING_DIMENSIONS") or os.getenv("VECTOR_DIMENSIONS")
    vector_dimensions = int(vector_dimensions_str.strip()) if vector_dimensions_str and vector_dimensions_str.strip() else None
    
    # HNSW index path
    vector_hnsw_index_path_str = os.getenv("VECTOR_HNSW_INDEX_PATH", "").strip()
    if not vector_hnsw_index_path_str:
        raise ValueError("VECTOR_HNSW_INDEX_PATH is required. Set it in .env file.")
    vector_hnsw_index_path = REPO_ROOT / vector_hnsw_index_path_str
    
    # Retrieval parameters
    retrieval_top_k_str = os.getenv("RETRIEVAL_TOP_K", "50").strip()
    retrieval_top_k = int(retrieval_top_k_str) if retrieval_top_k_str else 50
    
    use_rrf_str = os.getenv("USE_RRF", "True").strip().lower()
    use_rrf = use_rrf_str in ("true", "1", "yes")
    
    rrf_k_str = os.getenv("RRF_K", "60").strip()
    rrf_k = int(rrf_k_str) if rrf_k_str else 60
    
    rrf_top_k_str = os.getenv("RRF_TOP_K", "50").strip()
    rrf_top_k = int(rrf_top_k_str) if rrf_top_k_str else 50
    
    final_top_k_1_str = os.getenv("FINAL_TOP_K_1", "5").strip()
    final_top_k_1 = int(final_top_k_1_str) if final_top_k_1_str else 5
    
    final_top_k_2_str = os.getenv("FINAL_TOP_K_2", "10").strip()
    final_top_k_2 = int(final_top_k_2_str) if final_top_k_2_str else 10
    
    # Reranker parameters (not used for analysis, but required for initialization)
    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
    reranker_device = os.getenv("RERANKER_DEVICE", "mps").strip() or None
    reranker_batch_size_str = os.getenv("RERANKER_BATCH_SIZE", "32").strip()
    reranker_batch_size = int(reranker_batch_size_str) if reranker_batch_size_str else 32
    
    # BM25 cache
    bm25_cache_dir_str = os.getenv("BM25_CACHE_DIR", "").strip()
    bm25_cache_dir = REPO_ROOT / bm25_cache_dir_str if bm25_cache_dir_str else None
    bm25_cache_enabled_str = os.getenv("BM25_CACHE_ENABLED", "True").strip().lower()
    bm25_cache_enabled = bm25_cache_enabled_str in ("true", "1", "yes")
    
    # Vector parameters (for HybridRetriever initialization)
    vector_max_tokens_per_request = int(os.getenv("VECTOR_MAX_TOKENS_PER_REQUEST", "8192").strip())
    vector_max_items_per_batch_str = os.getenv("VECTOR_MAX_ITEMS_PER_BATCH", "").strip()
    vector_max_items_per_batch = int(vector_max_items_per_batch_str) if vector_max_items_per_batch_str else None
    vector_rpm_limit = int(os.getenv("VECTOR_RPM_LIMIT", "300").strip())
    vector_timeout = float(os.getenv("VECTOR_TIMEOUT", "120.0").strip())
    vector_normalize_embeddings_str = os.getenv("VECTOR_NORMALIZE_EMBEDDINGS", "True").strip().lower()
    vector_normalize_embeddings = vector_normalize_embeddings_str in ("true", "1", "yes")
    vector_use_hnsw_str = os.getenv("VECTOR_USE_HNSW", "True").strip().lower()
    vector_use_hnsw = vector_use_hnsw_str in ("true", "1", "yes")
    vector_hnsw_m = int(os.getenv("VECTOR_HNSW_M", "32").strip())
    vector_hnsw_ef_construction = int(os.getenv("VECTOR_HNSW_EF_CONSTRUCTION", "100").strip())
    vector_hnsw_ef_search = int(os.getenv("VECTOR_HNSW_EF_SEARCH", "64").strip())
    vector_embeddings_dir_str = os.getenv("VECTOR_EMBEDDINGS_DIR", "").strip()
    vector_embeddings_dir = REPO_ROOT / vector_embeddings_dir_str if vector_embeddings_dir_str else None
    vector_cache_embeddings_str = os.getenv("VECTOR_CACHE_EMBEDDINGS", "True").strip().lower()
    vector_cache_embeddings = vector_cache_embeddings_str in ("true", "1", "yes")
    vector_query_embedding_model = os.getenv("VECTOR_QUERY_EMBEDDING_MODEL", "").strip() or None
    
    return {
        'bm25_k1': bm25_k1,
        'bm25_b': bm25_b,
        'vector_local_model_name': vector_local_model_name,
        'vector_api_base': vector_api_base,
        'vector_api_key': vector_api_key,
        'vector_model_name': vector_model_name,
        'vector_dimensions': vector_dimensions,
        'vector_hnsw_index_path': vector_hnsw_index_path,
        'retrieval_top_k': retrieval_top_k,
        'use_rrf': use_rrf,
        'rrf_k': rrf_k,
        'rrf_top_k': rrf_top_k,
        'final_top_k_1': final_top_k_1,
        'final_top_k_2': final_top_k_2,
        'reranker_model': reranker_model,
        'reranker_device': reranker_device,
        'reranker_batch_size': reranker_batch_size,
        'bm25_cache_dir': bm25_cache_dir,
        'bm25_cache_enabled': bm25_cache_enabled,
        'vector_max_tokens_per_request': vector_max_tokens_per_request,
        'vector_max_items_per_batch': vector_max_items_per_batch,
        'vector_rpm_limit': vector_rpm_limit,
        'vector_timeout': vector_timeout,
        'vector_normalize_embeddings': vector_normalize_embeddings,
        'vector_use_hnsw': vector_use_hnsw,
        'vector_hnsw_m': vector_hnsw_m,
        'vector_hnsw_ef_construction': vector_hnsw_ef_construction,
        'vector_hnsw_ef_search': vector_hnsw_ef_search,
        'vector_embeddings_dir': vector_embeddings_dir,
        'vector_cache_embeddings': vector_cache_embeddings,
        'vector_query_embedding_model': vector_query_embedding_model,
    }


def print_statistics(stats: Optional[Dict], title: str):
    """Print statistics in a formatted table."""
    if stats is None:
        print(f"\n{title}: No data available")
        return
    
    print(f"\n{title}:")
    print(f"  Count:      {stats['count']}")
    print(f"  Min:        {stats['min']:.6f}")
    print(f"  Max:        {stats['max']:.6f}")
    print(f"  Mean:       {stats['mean']:.6f}")
    print(f"  Median:     {stats['median']:.6f}")
    print(f"  P25:        {stats['p25']:.6f}")
    print(f"  P75:        {stats['p75']:.6f}")
    print(f"  P90:        {stats['p90']:.6f}")
    print(f"  P95:        {stats['p95']:.6f}")


def print_recommendations(
    combined_stats: Optional[Dict],
    diff_stats: Optional[Dict],
    use_rrf: bool,
):
    """Print recommended threshold values based on analysis."""
    print("\n" + "=" * 80)
    print("Recommended Thresholds")
    print("=" * 80)
    
    if combined_stats is None:
        print("\n⚠️  Insufficient data for recommendations")
        return
    
    # Recommend MIN_SCORE based on percentiles
    # Use P75 as conservative threshold, P90 as aggressive threshold
    p75_score = combined_stats['p75']
    p90_score = combined_stats['p90']
    
    # Recommend SCORE_DIFF_THRESHOLD based on difference statistics
    if diff_stats:
        mean_diff = diff_stats['mean']
        median_diff = diff_stats['median']
        p90_diff = diff_stats['p90']
        # Use 2-3x median or mean as threshold
        recommended_diff = max(mean_diff * 2, median_diff * 2.5)
    else:
        # Fallback: use 10% of score range as threshold
        score_range = combined_stats['max'] - combined_stats['min']
        recommended_diff = score_range * 0.1
    
    print("\nBased on score distribution analysis:")
    print(f"\n1. RERANKER_PREFILTER_MIN_SCORE")
    print(f"   Conservative (P75): {p75_score:.6f}")
    print(f"   Aggressive (P90):  {p90_score:.6f}")
    print(f"   Recommended:       {p75_score:.6f}")
    
    print(f"\n2. RERANKER_PREFILTER_SCORE_DIFF_THRESHOLD")
    print(f"   Based on median diff (2.5x): {diff_stats['median'] * 2.5:.6f}" if diff_stats else "   (insufficient data)")
    print(f"   Based on mean diff (2x):     {diff_stats['mean'] * 2:.6f}" if diff_stats else "   (insufficient data)")
    print(f"   Recommended:                 {recommended_diff:.6f}")
    
    print(f"\n3. RERANKER_PREFILTER_MIN_ITEMS")
    print(f"   Recommended: 20 (default)")
    
    print("\n" + "=" * 80)
    print("Suggested .env configuration:")
    print("=" * 80)
    print(f"RERANKER_PREFILTER_ENABLED=True")
    print(f"RERANKER_PREFILTER_MIN_SCORE={p75_score:.6f}")
    print(f"RERANKER_PREFILTER_SCORE_DIFF_THRESHOLD={recommended_diff:.6f}")
    print(f"RERANKER_PREFILTER_MIN_ITEMS=20")
    print("=" * 80)


def main():
    """Main analysis function."""
    # Load data paths
    data_dir = REPO_ROOT / "data" / "test"
    items_path = data_dir / "500_items.csv"
    queries_path = data_dir / "10_queries.csv"

    if not items_path.exists() or not queries_path.exists():
        print(f"❌ Error: Data files not found")
        print(f"   Expected: {items_path}")
        print(f"   Expected: {queries_path}")
        sys.exit(1)

    print("=" * 80)
    print("Score Distribution Analysis")
    print("=" * 80)

    # Load candidates
    print("\nLoading candidates...")
    config = load_retriever_config()
    
    candidates, data_hash = load_food_candidates_for_hybrid(
        items_path,
        cache_dir=config['bm25_cache_dir'],
        cache_enabled=config['bm25_cache_enabled'],
    )
    print(f"  ✓ Loaded {len(candidates)} candidates")

    # Load queries
    queries = []
    with queries_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        print(f"  Debug: CSV fieldnames = {fieldnames}")
        
        row_count = 0
        for row_num, row in enumerate(reader, start=2):  # Start from 2 (1 is header)
            row_count += 1
            # Try both "id" and "query_id" for query ID
            query_id = row.get("query_id") or row.get("id", "")
            # Try both "search_term_pt" and "search_term" for query text
            query_text = row.get("search_term_pt") or row.get("search_term", "")
            
            if query_id and query_text:
                queries.append((str(query_id), str(query_text)))
            else:
                print(f"  Debug: Row {row_num} skipped - query_id='{query_id}', query_text='{query_text[:50] if query_text else ''}...'")
        
        print(f"  Debug: Processed {row_count} data rows")

    if not queries:
        print(f"\n  ❌ Error: No queries loaded from {queries_path}")
        if queries_path.exists():
            with queries_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                print(f"     CSV headers found: {headers}")
                # Show first data row
                try:
                    first_row = next(reader)
                    print(f"     First data row sample: {first_row}")
                except StopIteration:
                    print(f"     CSV file appears to be empty (no data rows)")
        else:
            print(f"     File does not exist")
        print(f"     Expected columns: 'id' or 'query_id', 'search_term_pt' or 'search_term'")
        sys.exit(1)
    
    print(f"  ✓ Loaded {len(queries)} queries")

    # Initialize retriever with pre-filter DISABLED
    print("\nInitializing hybrid retriever (pre-filter disabled for analysis)...")
    try:
        retriever = HybridRetriever(
            candidates,
            bm25_k1=config['bm25_k1'],
            bm25_b=config['bm25_b'],
            vector_hnsw_index_path=config['vector_hnsw_index_path'],
            retrieval_top_k=config['retrieval_top_k'],
            use_rrf=config['use_rrf'],
            rrf_k=config['rrf_k'],
            rrf_top_k=config['rrf_top_k'],
            final_top_k_1=config['final_top_k_1'],
            final_top_k_2=config['final_top_k_2'],
            reranker_model=config['reranker_model'],
            reranker_device=config['reranker_device'],
            reranker_batch_size=config['reranker_batch_size'],
            # Disable pre-filter for analysis
            reranker_prefilter_enabled=False,
            reranker_prefilter_min_score=None,
            reranker_prefilter_score_diff_threshold=None,
            reranker_prefilter_min_items=20,
            # Vector parameters
            vector_api_base=config['vector_api_base'],
            vector_api_key=config['vector_api_key'],
            vector_model_name=config['vector_model_name'],
            vector_dimensions=config['vector_dimensions'],
            vector_local_model_name=config['vector_local_model_name'],
            bm25_cache_dir=config['bm25_cache_dir'],
            bm25_cache_enabled=config['bm25_cache_enabled'],
            bm25_data_source_hash=data_hash,
        )
        print("  ✓ Hybrid retriever initialized")
    except Exception as e:
        print(f"  ❌ Failed to initialize retriever: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Collect scores from all queries
    all_bm25_scores: List[float] = []
    all_vector_scores: List[float] = []
    all_combined_scores: List[float] = []
    all_score_diffs: List[float] = []

    print("\nAnalyzing scores from queries...")
    num_queries = min(len(queries), 10)  # Analyze up to 10 queries
    for i, (query_id, query_text) in enumerate(queries[:num_queries], 1):
        print(f"  [{i}/{num_queries}] Processing query {query_id}...")
        try:
            # Execute search (pre-filter is disabled)
            result = retriever.search(query_text, query_id=query_id)

            # Collect BM25 and Vector scores
            all_bm25_scores.extend([score for _, score in result.bm25_results])
            all_vector_scores.extend([score for _, score in result.vector_results])

            # Calculate combined scores (RRF or max) for merged items
            merged_item_ids = {item.id for item in result.merged_items}
            for item_id in merged_item_ids:
                combined_score = calculate_combined_score(
                    item_id,
                    result.bm25_results,
                    result.vector_results,
                    config['use_rrf'],
                    config['rrf_k'],
                )
                all_combined_scores.append(combined_score)

            # Calculate score differences for combined scores (sorted)
            if len(result.merged_items) > 1:
                # Get combined scores for merged items, sorted
                merged_scores = [
                    calculate_combined_score(
                        item.id,
                        result.bm25_results,
                        result.vector_results,
                        config['use_rrf'],
                        config['rrf_k'],
                    )
                    for item in result.merged_items
                ]
                merged_scores.sort(reverse=True)
                
                # Calculate differences
                for j in range(len(merged_scores) - 1):
                    diff = merged_scores[j] - merged_scores[j + 1]
                    all_score_diffs.append(diff)

        except Exception as e:
            print(f"    ⚠ Error processing query {query_id}: {e}")
            continue

    # Analyze scores
    print("\n" + "=" * 80)
    print("Score Distribution Statistics")
    print("=" * 80)

    # Create result lists for analysis
    bm25_results = list(zip([None] * len(all_bm25_scores), all_bm25_scores))
    vector_results = list(zip([None] * len(all_vector_scores), all_vector_scores))
    combined_results = list(zip([None] * len(all_combined_scores), all_combined_scores))
    diff_results = list(zip([None] * len(all_score_diffs), all_score_diffs)) if all_score_diffs else []

    # Analyze each score type
    bm25_stats = analyze_scores(bm25_results, "BM25")
    vector_stats = analyze_scores(vector_results, "Vector")
    combined_stats = analyze_scores(combined_results, "RRF" if config['use_rrf'] else "Max(BM25,Vector)")
    diff_stats = analyze_score_differences(combined_results) if combined_results else None

    print_statistics(bm25_stats, "BM25 Scores")
    print_statistics(vector_stats, "Vector Scores")
    print_statistics(combined_stats, f"Combined Scores ({'RRF' if config['use_rrf'] else 'Max'})")
    
    if diff_stats:
        print("\nAdjacent Score Differences:")
        print(f"  Count:        {diff_stats['count']}")
        print(f"  Min:          {diff_stats['min']:.6f}")
        print(f"  Max:          {diff_stats['max']:.6f}")
        print(f"  Mean:         {diff_stats['mean']:.6f}")
        print(f"  Median:       {diff_stats['median']:.6f}")
        print(f"  P90:          {diff_stats['p90']:.6f}")
        print(f"  P95:          {diff_stats['p95']:.6f}")
        print(f"  Large drops:  {diff_stats['large_drops']} (diff > 3x median)")

    # Print recommendations
    print_recommendations(combined_stats, diff_stats, config['use_rrf'])

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print("\nNote: These recommendations are based on the analyzed queries.")
    print("      You may need to adjust thresholds based on your specific use case.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
