#!/usr/bin/env python3
"""Demo script for hybrid search with query selector.

This script allows users to select a query from data/raw/queries.csv (100 queries)
and perform hybrid search, displaying:
- Total latency
- Per-stage timing breakdown
- Top-10 results

Usage:
    python scripts/demo_query_selector.py [query_index]
    
    query_index: Optional. Row number (1-100) to select. If not provided, will prompt interactively.
    
Example:
    python scripts/demo_query_selector.py 5  # Select query at row 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path for module imports
REPO_ROOT = Path(__file__).resolve().parents[1]
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

from src.hybrid_retrieval_no_llm import HybridRetriever, load_food_candidates_for_hybrid


def format_item_result(item, score: float, rank: int) -> str:
    """Format a single item result for display."""
    name = item.name[:60] + "..." if len(item.name) > 60 else item.name
    return f"  {rank:2d}. [{item.id}] {name} (score: {score:.4f})"


def load_queries(queries_path: Path) -> list[dict]:
    """Load all queries from CSV file.
    
    Returns:
        List of dictionaries with 'id' and 'search_term_pt' keys
    """
    queries = []
    with queries_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                'id': row.get('id', '').strip(),
                'search_term_pt': row.get('search_term_pt', '').strip()
            })
    return queries


def select_query_interactive(queries: list[dict]) -> dict:
    """Interactively select a query from the list."""
    print("\n" + "=" * 80)
    print("Available Queries (showing first 10)")
    print("=" * 80)
    for i, q in enumerate(queries[:10], 1):
        print(f"  {i:3d}. [{q['id']}] {q['search_term_pt']}")
    if len(queries) > 10:
        print(f"  ... (and {len(queries) - 10} more queries)")
    
    while True:
        try:
            user_input = input(f"\nEnter query number (1-{len(queries)}): ").strip()
            query_index = int(user_input)
            if 1 <= query_index <= len(queries):
                return queries[query_index - 1]
            else:
                print(f"❌ Invalid input. Please enter a number between 1 and {len(queries)}.")
        except ValueError:
            print(f"❌ Invalid input. Please enter a number between 1 and {len(queries)}.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled by user.")
            sys.exit(0)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Demo hybrid search with query selector",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'query_index',
        type=int,
        nargs='?',
        help=f'Query row number (1-100) to select. If not provided, will prompt interactively.'
    )
    parser.add_argument(
        '--items',
        type=str,
        default='data/raw/5k_items.csv',
        help='Path to items CSV file (relative to repo root). Default: data/raw/5k_items.csv'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    queries_path = REPO_ROOT / "data" / "raw" / "queries.csv"
    items_path = REPO_ROOT / args.items
    
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    
    print("=" * 80)
    print("Hybrid Search Demo - Query Selector")
    print("=" * 80)
    
    # Load queries
    print(f"\nLoading queries from: {queries_path}")
    queries = load_queries(queries_path)
    print(f"  ✓ Loaded {len(queries)} queries")
    
    # Select query
    if args.query_index:
        if 1 <= args.query_index <= len(queries):
            selected_query = queries[args.query_index - 1]
            print(f"\n  ✓ Selected query #{args.query_index}")
        else:
            print(f"\n❌ Error: Query index {args.query_index} is out of range (1-{len(queries)})")
            sys.exit(1)
    else:
        selected_query = select_query_interactive(queries)
    
    query_id = selected_query['id']
    query_text = selected_query['search_term_pt']
    
    print(f"\nSelected Query:")
    print(f"  ID: {query_id}")
    print(f"  Text: {query_text}")
    
    # Load configuration from environment variables (same as original demo)
    print("\nLoading configuration from environment variables...")
    
    # BM25 caching parameters
    bm25_cache_dir_str = os.getenv("BM25_CACHE_DIR")
    bm25_cache_dir = None
    if bm25_cache_dir_str and bm25_cache_dir_str.strip():
        bm25_cache_dir = REPO_ROOT / bm25_cache_dir_str.strip()
    
    bm25_cache_enabled_str = os.getenv("BM25_CACHE_ENABLED", "True").strip().lower()
    bm25_cache_enabled = bm25_cache_enabled_str in ("true", "1", "yes")
    
    # Load candidates
    print(f"\nLoading candidates from: {items_path}")
    candidates, data_source_hash = load_food_candidates_for_hybrid(
        items_path,
        cache_dir=bm25_cache_dir,
        cache_enabled=bm25_cache_enabled,
    )
    print(f"  ✓ Loaded {len(candidates)} items")
    
    # BM25 parameters
    bm25_k1_str = os.getenv("BM25_K1")
    if not bm25_k1_str or not bm25_k1_str.strip():
        raise ValueError("BM25_K1 is required. Set it in .env file or environment variable (e.g., '1.5').")
    bm25_k1 = float(bm25_k1_str.strip())
    
    bm25_b_str = os.getenv("BM25_B")
    if not bm25_b_str or not bm25_b_str.strip():
        raise ValueError("BM25_B is required. Set it in .env file or environment variable (e.g., '0.75').")
    bm25_b = float(bm25_b_str.strip())
    
    # Vector retrieval parameters - Local model (preferred)
    vector_local_model_name = os.getenv("VECTOR_LOCAL_MODEL_NAME")
    if vector_local_model_name:
        vector_local_model_name = vector_local_model_name.strip()
        if not vector_local_model_name:
            vector_local_model_name = None
    else:
        vector_local_model_name = None
    
    # Validate: must have local model config
    if not vector_local_model_name:
        print("\n" + "=" * 80)
        print("ERROR: Missing vector retrieval configuration!")
        print("=" * 80)
        print("Please set VECTOR_LOCAL_MODEL_NAME in your .env file:")
        print("   VECTOR_LOCAL_MODEL_NAME=BAAI/bge-m3")
        print("=" * 80)
        sys.exit(1)
    
    # Vector parameters
    vector_normalize_embeddings_str = os.getenv("VECTOR_NORMALIZE_EMBEDDINGS", "True").strip().lower()
    vector_normalize_embeddings = vector_normalize_embeddings_str in ("true", "1", "yes")
    
    # HNSW index path
    vector_hnsw_index_path_str = os.getenv("VECTOR_HNSW_INDEX_PATH")
    if not vector_hnsw_index_path_str or not vector_hnsw_index_path_str.strip():
        raise ValueError(
            "VECTOR_HNSW_INDEX_PATH is required. Set it in .env file or environment variable.\n"
            "This must be the full path to an existing .index file."
        )
    vector_hnsw_index_path = REPO_ROOT / vector_hnsw_index_path_str.strip()
    
    if not vector_hnsw_index_path.exists():
        raise ValueError(f"HNSW index file does not exist: {vector_hnsw_index_path}")
    
    vector_use_hnsw_str = os.getenv("VECTOR_USE_HNSW", "True").strip().lower()
    vector_use_hnsw = vector_use_hnsw_str in ("true", "1", "yes")
    
    vector_hnsw_m = int(os.getenv("VECTOR_HNSW_M", "32").strip())
    vector_hnsw_ef_construction = int(os.getenv("VECTOR_HNSW_EF_CONSTRUCTION", "100").strip())
    vector_hnsw_ef_search = int(os.getenv("VECTOR_HNSW_EF_SEARCH", "64").strip())
    
    vector_embeddings_dir_str = os.getenv("VECTOR_EMBEDDINGS_DIR")
    vector_embeddings_dir = None
    if vector_embeddings_dir_str and vector_embeddings_dir_str.strip():
        vector_embeddings_dir = REPO_ROOT / vector_embeddings_dir_str.strip()
    
    vector_cache_embeddings_str = os.getenv("VECTOR_CACHE_EMBEDDINGS", "True").strip().lower()
    vector_cache_embeddings = vector_cache_embeddings_str in ("true", "1", "yes")
    
    vector_query_embedding_model = os.getenv("VECTOR_QUERY_EMBEDDING_MODEL")
    if vector_query_embedding_model:
        vector_query_embedding_model = vector_query_embedding_model.strip()
        if not vector_query_embedding_model:
            vector_query_embedding_model = None
    
    # Retrieval parameters
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "50").strip())
    
    # RRF fusion parameters
    use_rrf_str = os.getenv("USE_RRF", "True").strip().lower()
    use_rrf = use_rrf_str in ("true", "1", "yes")
    
    rrf_k = int(os.getenv("RRF_K", "60").strip())
    rrf_top_k = int(os.getenv("RRF_TOP_K", "20").strip())
    
    # Final output parameters
    final_top_k_1 = int(os.getenv("FINAL_TOP_K_1", "5").strip())
    final_top_k_2 = int(os.getenv("FINAL_TOP_K_2", "10").strip())
    
    # Cross-Encoder re-ranking parameters
    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
    reranker_device = os.getenv("RERANKER_DEVICE", "mps").strip() or None
    reranker_batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "32").strip())
    reranker_top_k_str = os.getenv("RERANKER_TOP_K")
    reranker_top_k = None
    if reranker_top_k_str and reranker_top_k_str.strip():
        reranker_top_k = int(reranker_top_k_str.strip())
    
    reranker_tokenization_cache_dir_str = os.getenv("RERANKER_TOKENIZATION_CACHE_DIR")
    reranker_tokenization_cache_dir = None
    if reranker_tokenization_cache_dir_str and reranker_tokenization_cache_dir_str.strip():
        reranker_tokenization_cache_dir = REPO_ROOT / reranker_tokenization_cache_dir_str.strip()
    
    reranker_tokenization_cache_enabled_str = os.getenv("RERANKER_TOKENIZATION_CACHE_ENABLED", "True").strip().lower()
    reranker_tokenization_cache_enabled = reranker_tokenization_cache_enabled_str in ("true", "1", "yes")
    
    reranker_max_concurrent_batches = int(os.getenv("RERANKER_MAX_CONCURRENT_BATCHES", "2").strip())
    
    # Pre-filtering parameters
    reranker_prefilter_enabled_str = os.getenv("RERANKER_PREFILTER_ENABLED", "True").strip().lower()
    reranker_prefilter_enabled = reranker_prefilter_enabled_str in ("true", "1", "yes")
    
    reranker_prefilter_min_score_str = os.getenv("RERANKER_PREFILTER_MIN_SCORE")
    reranker_prefilter_min_score = None
    if reranker_prefilter_min_score_str and reranker_prefilter_min_score_str.strip():
        reranker_prefilter_min_score = float(reranker_prefilter_min_score_str.strip())
    
    reranker_prefilter_score_diff_threshold_str = os.getenv("RERANKER_PREFILTER_SCORE_DIFF_THRESHOLD")
    reranker_prefilter_score_diff_threshold = None
    if reranker_prefilter_score_diff_threshold_str and reranker_prefilter_score_diff_threshold_str.strip():
        reranker_prefilter_score_diff_threshold = float(reranker_prefilter_score_diff_threshold_str.strip())
    
    reranker_prefilter_min_items = int(os.getenv("RERANKER_PREFILTER_MIN_ITEMS", "20").strip())
    
    # Initialize hybrid retriever
    print("\nInitializing hybrid retriever...")
    retriever = HybridRetriever(
        candidates,
        # BM25 parameters
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        # Required parameters
        vector_hnsw_index_path=vector_hnsw_index_path,
        retrieval_top_k=retrieval_top_k,
        use_rrf=use_rrf,
        rrf_k=rrf_k,
        rrf_top_k=rrf_top_k,
        final_top_k_1=final_top_k_1,
        final_top_k_2=final_top_k_2,
        reranker_model=reranker_model,
        # Local model parameters
        vector_local_model_name=vector_local_model_name,
        vector_normalize_embeddings=vector_normalize_embeddings,
        vector_use_hnsw=vector_use_hnsw,
        vector_hnsw_m=vector_hnsw_m,
        vector_hnsw_ef_construction=vector_hnsw_ef_construction,
        vector_hnsw_ef_search=vector_hnsw_ef_search,
        vector_embeddings_dir=vector_embeddings_dir,
        vector_cache_embeddings=vector_cache_embeddings,
        vector_query_embedding_model=vector_query_embedding_model,
        # Cross-Encoder re-ranking parameters
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
        # BM25 caching parameters
        bm25_cache_dir=bm25_cache_dir,
        bm25_cache_enabled=bm25_cache_enabled,
        bm25_data_source_hash=data_source_hash,
    )
    print("  ✓ Hybrid retriever initialized")
    
    # Perform hybrid retrieval
    print("\n" + "=" * 80)
    print("Performing Hybrid Search...")
    print("=" * 80)
    print(f"Query: {query_text}")
    print(f"Query ID: {query_id}")
    
    result = retriever.search(query_text, query_id=query_id)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Top-10 Results
    print("\nTop-10 Results:")
    print("-" * 80)
    if result.top_10:
        for rank, (candidate, score) in enumerate(result.top_10, 1):
            print(format_item_result(candidate, score, rank))
        print(f"\nScore Range: {min(s for _, s in result.top_10):.4f} - {max(s for _, s in result.top_10):.4f}")
    else:
        print("  No results found.")
    
    # Timing information
    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN")
    print("=" * 80)
    print(f"  BM25 Retrieval:        {result.bm25_time*1000:.2f} ms")
    print(f"  Vector Retrieval:      {result.vector_time*1000:.2f} ms")
    if use_rrf:
        print(f"  RRF Fusion:            {result.rrf_time*1000:.2f} ms")
    else:
        print(f"  Merge & Deduplicate:   {result.rrf_time*1000:.2f} ms")
    print(f"  Cross-Encoder Re-rank: {result.rerank_time*1000:.2f} ms")
    print("-" * 80)
    print(f"  TOTAL LATENCY:         {result.total_time*1000:.2f} ms ({result.total_time:.3f} s)")
    print("=" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Query ID: {query_id}")
    print(f"  Query: {query_text}")
    print(f"  BM25 Results: {len(result.bm25_results)} items")
    print(f"  Vector Results: {len(result.vector_results)} items")
    print(f"  Merged Items: {len(result.merged_items)} unique items")
    print(f"  Top-10 Results: {len(result.top_10)} items")
    print(f"  Total Latency: {result.total_time*1000:.2f} ms ({result.total_time:.3f} s)")
    print("=" * 80)
    
    print("\n✓ Demo completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

