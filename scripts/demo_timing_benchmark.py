#!/usr/bin/env python3
"""Benchmark script for measuring query timing at different stages.

This script focuses on measuring query performance across different stages:
1. BM25 retrieval
2. Vector retrieval
3. RRF fusion / merge
4. Cross-Encoder re-ranking
5. Total time

Usage:
    python scripts/demo_timing_benchmark.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path for module imports
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    # Load .env file from project root directory
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try to load from current directory or default location
        load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

import logging

# Configure logging to show INFO level messages (needed to see reranker scores)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

from src.hybrid_retrieval_no_llm import HybridRetriever, load_food_candidates_for_hybrid


def format_item_result(item, score: float, rank: int) -> str:
    """Format a single item result for display."""
    name = item.name[:60] + "..." if len(item.name) > 60 else item.name
    return f"  {rank:2d}. [{item.id}] {name} (score: {score:.4f})"


def load_queries(queries_path: Path) -> List[tuple[str, str]]:
    """Load queries from CSV file.
    
    Returns:
        List of (query_id, query_text) tuples.
    """
    queries = []
    if not queries_path.exists():
        return queries
    
    with queries_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("id") or row.get("query_id") or f"query_{len(queries)}"
            query_text = row.get("search_term_pt") or row.get("query") or row.get("text") or ""
            if query_text.strip():
                queries.append((query_id, query_text.strip()))
    
    return queries


def main():
    """Benchmark hybrid search timing."""
    # Load data
    data_dir = REPO_ROOT / "data" / "raw"
    items_path = data_dir / "5k_items.csv"
    queries_path = data_dir / "queries.csv"
    
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    print("=" * 100)
    print("Hybrid Search Timing Benchmark")
    print("=" * 100)
    
    # Load queries
    print("\nLoading queries...")
    queries = load_queries(queries_path)
    if not queries:
        raise ValueError(f"No valid queries found in {queries_path}")
    print(f"  ✓ Loaded {len(queries)} queries")
    
    # Use only the first query for benchmarking
    query_id, query_text = queries[0]
    print(f"  Using query: {query_text[:60]}{'...' if len(query_text) > 60 else ''}")
    
    # Load .env file from project root
    print("\nLoading configuration...")
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        print(f"  ✓ Found .env file at: {env_path}")
    else:
        print(f"  ⚠ .env file not found at: {env_path}")
        print("     (Will try to read from environment variables)")
    
    print("\nLoading candidates...")
    
    # BM25 caching parameters (read before loading candidates)
    bm25_cache_dir_str = os.getenv("BM25_CACHE_DIR")
    bm25_cache_dir = None
    if bm25_cache_dir_str and bm25_cache_dir_str.strip():
        bm25_cache_dir = REPO_ROOT / bm25_cache_dir_str.strip()
    
    bm25_cache_enabled_str = os.getenv("BM25_CACHE_ENABLED", "True").strip().lower()
    bm25_cache_enabled = bm25_cache_enabled_str in ("true", "1", "yes")
    
    # Load candidates with caching support
    candidates, data_source_hash = load_food_candidates_for_hybrid(
        items_path,
        cache_dir=bm25_cache_dir,
        cache_enabled=bm25_cache_enabled,
    )
    print(f"  ✓ Loaded {len(candidates)} items")
    
    # Get all required parameters from environment variables (same as demo_hybrid_search_no_llm.py)
    print("\nLoading configuration from environment variables...")
    
    # BM25 parameters
    bm25_k1_str = os.getenv("BM25_K1")
    if not bm25_k1_str or not bm25_k1_str.strip():
        raise ValueError("BM25_K1 is required. Set it in .env file or environment variable (e.g., '1.5').")
    try:
        bm25_k1 = float(bm25_k1_str.strip())
        if bm25_k1 <= 0:
            raise ValueError(f"bm25_k1 must be positive, got {bm25_k1}")
    except ValueError as e:
        raise ValueError(f"Invalid BM25_K1 value '{bm25_k1_str}': {e}")
    
    bm25_b_str = os.getenv("BM25_B")
    if not bm25_b_str or not bm25_b_str.strip():
        raise ValueError("BM25_B is required. Set it in .env file or environment variable (e.g., '0.75').")
    try:
        bm25_b = float(bm25_b_str.strip())
        if bm25_b < 0 or bm25_b > 1:
            raise ValueError(f"bm25_b must be between 0 and 1, got {bm25_b}")
    except ValueError as e:
        raise ValueError(f"Invalid BM25_B value '{bm25_b_str}': {e}")
    
    # Vector retrieval parameters - choose one: Local model or API
    vector_local_model_name = os.getenv("VECTOR_LOCAL_MODEL_NAME")
    if vector_local_model_name:
        vector_local_model_name = vector_local_model_name.strip()
        if not vector_local_model_name:
            vector_local_model_name = None
    else:
        vector_local_model_name = None
    
    vector_api_base = os.getenv("VECTOR_API_BASE")
    if vector_api_base:
        vector_api_base = vector_api_base.strip()
        if not vector_api_base:
            vector_api_base = None
    else:
        vector_api_base = None
    
    vector_api_key = os.getenv("VECTOR_API_KEY") or os.getenv("OPENAI_API_KEY")
    if vector_api_key:
        vector_api_key = vector_api_key.strip()
        if not vector_api_key:
            vector_api_key = None
    else:
        vector_api_key = None
    
    if not vector_local_model_name and not vector_api_key:
        print("\n" + "=" * 100)
        print("ERROR: Missing vector retrieval configuration!")
        print("=" * 100)
        print("Please set one of the following in your .env file:")
        print("\n1. For local model (recommended, no API key needed):")
        print("   VECTOR_LOCAL_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("\n2. For API (requires API key):")
        print("   VECTOR_API_KEY=sk-your-key-here")
        print(f"\n   Expected .env location: {env_path}")
        print("=" * 100)
        return
    
    vector_model_name = os.getenv("VECTOR_MODEL_NAME")
    if vector_model_name:
        vector_model_name = vector_model_name.strip()
    else:
        vector_model_name = None
    
    vector_dimensions_str = os.getenv("VECTOR_DIMENSIONS")
    vector_dimensions = None
    if vector_dimensions_str and vector_dimensions_str.strip():
        try:
            vector_dimensions = int(vector_dimensions_str.strip())
            if vector_dimensions <= 0:
                raise ValueError(f"vector_dimensions must be positive, got {vector_dimensions}")
        except ValueError as e:
            raise ValueError(f"Invalid VECTOR_DIMENSIONS value '{vector_dimensions_str}': {e}")
    
    vector_max_tokens_per_request_str = os.getenv("VECTOR_MAX_TOKENS_PER_REQUEST", "8192").strip()
    try:
        vector_max_tokens_per_request = int(vector_max_tokens_per_request_str)
        if vector_max_tokens_per_request <= 0:
            raise ValueError(f"vector_max_tokens_per_request must be positive, got {vector_max_tokens_per_request}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_MAX_TOKENS_PER_REQUEST value '{vector_max_tokens_per_request_str}': {e}")
    
    vector_max_items_per_batch_str = os.getenv("VECTOR_MAX_ITEMS_PER_BATCH")
    vector_max_items_per_batch = None
    if vector_max_items_per_batch_str and vector_max_items_per_batch_str.strip():
        try:
            vector_max_items_per_batch = int(vector_max_items_per_batch_str.strip())
            if vector_max_items_per_batch <= 0:
                raise ValueError(f"vector_max_items_per_batch must be positive, got {vector_max_items_per_batch}")
        except ValueError as e:
            raise ValueError(f"Invalid VECTOR_MAX_ITEMS_PER_BATCH value '{vector_max_items_per_batch_str}': {e}")
    
    vector_rpm_limit_str = os.getenv("VECTOR_RPM_LIMIT", "300").strip()
    try:
        vector_rpm_limit = int(vector_rpm_limit_str)
        if vector_rpm_limit <= 0:
            raise ValueError(f"vector_rpm_limit must be positive, got {vector_rpm_limit}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_RPM_LIMIT value '{vector_rpm_limit_str}': {e}")
    
    vector_timeout_str = os.getenv("VECTOR_TIMEOUT", "120.0").strip()
    try:
        vector_timeout = float(vector_timeout_str)
        if vector_timeout <= 0:
            raise ValueError(f"vector_timeout must be positive, got {vector_timeout}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_TIMEOUT value '{vector_timeout_str}': {e}")
    
    vector_normalize_embeddings_str = os.getenv("VECTOR_NORMALIZE_EMBEDDINGS", "True").strip().lower()
    vector_normalize_embeddings = vector_normalize_embeddings_str in ("true", "1", "yes")
    
    vector_hnsw_index_path_str = os.getenv("VECTOR_HNSW_INDEX_PATH")
    if not vector_hnsw_index_path_str or not vector_hnsw_index_path_str.strip():
        raise ValueError(
            "VECTOR_HNSW_INDEX_PATH is required. Set it in .env file or environment variable.\n"
            "This must be the full path to an existing .index file (e.g., 'data/vector_indices/hnsw_text_embedding_3_small_dim1536_500.index').\n"
            "Please generate the index file first using vector_retrieval.py"
        )
    vector_hnsw_index_path = REPO_ROOT / vector_hnsw_index_path_str.strip()
    
    if not vector_hnsw_index_path.exists():
        raise ValueError(
            f"HNSW index file does not exist: {vector_hnsw_index_path}\n"
            f"Please generate the index file first using vector_retrieval.py"
        )
    
    vector_use_hnsw_str = os.getenv("VECTOR_USE_HNSW", "True").strip().lower()
    vector_use_hnsw = vector_use_hnsw_str in ("true", "1", "yes")
    
    vector_hnsw_m_str = os.getenv("VECTOR_HNSW_M")
    if not vector_hnsw_m_str or not vector_hnsw_m_str.strip():
        raise ValueError("VECTOR_HNSW_M is required. Set it in .env file or environment variable (e.g., '32').")
    try:
        vector_hnsw_m = int(vector_hnsw_m_str.strip())
        if vector_hnsw_m <= 0:
            raise ValueError(f"vector_hnsw_m must be positive, got {vector_hnsw_m}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_HNSW_M value '{vector_hnsw_m_str}': {e}")
    
    vector_hnsw_ef_construction_str = os.getenv("VECTOR_HNSW_EF_CONSTRUCTION")
    if not vector_hnsw_ef_construction_str or not vector_hnsw_ef_construction_str.strip():
        raise ValueError("VECTOR_HNSW_EF_CONSTRUCTION is required. Set it in .env file or environment variable (e.g., '100').")
    try:
        vector_hnsw_ef_construction = int(vector_hnsw_ef_construction_str.strip())
        if vector_hnsw_ef_construction <= 0:
            raise ValueError(f"vector_hnsw_ef_construction must be positive, got {vector_hnsw_ef_construction}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_HNSW_EF_CONSTRUCTION value '{vector_hnsw_ef_construction_str}': {e}")
    
    vector_hnsw_ef_search_str = os.getenv("VECTOR_HNSW_EF_SEARCH")
    if not vector_hnsw_ef_search_str or not vector_hnsw_ef_search_str.strip():
        raise ValueError("VECTOR_HNSW_EF_SEARCH is required. Set it in .env file or environment variable (e.g., '64').")
    try:
        vector_hnsw_ef_search = int(vector_hnsw_ef_search_str.strip())
        if vector_hnsw_ef_search <= 0:
            raise ValueError(f"vector_hnsw_ef_search must be positive, got {vector_hnsw_ef_search}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_HNSW_EF_SEARCH value '{vector_hnsw_ef_search_str}': {e}")
    
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
    
    retrieval_top_k_str = os.getenv("RETRIEVAL_TOP_K")
    if not retrieval_top_k_str or not retrieval_top_k_str.strip():
        raise ValueError("RETRIEVAL_TOP_K is required. Set it in .env file or environment variable (e.g., '50').")
    try:
        retrieval_top_k = int(retrieval_top_k_str.strip())
        if retrieval_top_k <= 0:
            raise ValueError(f"retrieval_top_k must be positive, got {retrieval_top_k}")
    except ValueError as e:
        raise ValueError(f"Invalid RETRIEVAL_TOP_K value '{retrieval_top_k_str}': {e}")
    
    use_rrf_str = os.getenv("USE_RRF", "True").strip().lower()
    use_rrf = use_rrf_str in ("true", "1", "yes")
    
    rrf_k_str = os.getenv("RRF_K")
    if not rrf_k_str or not rrf_k_str.strip():
        raise ValueError("RRF_K is required. Set it in .env file or environment variable (e.g., '60').")
    try:
        rrf_k = int(rrf_k_str.strip())
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
    except ValueError as e:
        raise ValueError(f"Invalid RRF_K value '{rrf_k_str}': {e}")
    
    rrf_top_k_str = os.getenv("RRF_TOP_K")
    if not rrf_top_k_str or not rrf_top_k_str.strip():
        raise ValueError("RRF_TOP_K is required. Set it in .env file or environment variable (e.g., '50').")
    try:
        rrf_top_k = int(rrf_top_k_str.strip())
        if rrf_top_k <= 0:
            raise ValueError(f"rrf_top_k must be positive, got {rrf_top_k}")
    except ValueError as e:
        raise ValueError(f"Invalid RRF_TOP_K value '{rrf_top_k_str}': {e}")
    
    final_top_k_1_str = os.getenv("FINAL_TOP_K_1")
    if not final_top_k_1_str or not final_top_k_1_str.strip():
        raise ValueError("FINAL_TOP_K_1 is required. Set it in .env file or environment variable (e.g., '5').")
    try:
        final_top_k_1 = int(final_top_k_1_str.strip())
        if final_top_k_1 <= 0:
            raise ValueError(f"final_top_k_1 must be positive, got {final_top_k_1}")
    except ValueError as e:
        raise ValueError(f"Invalid FINAL_TOP_K_1 value '{final_top_k_1_str}': {e}")
    
    final_top_k_2_str = os.getenv("FINAL_TOP_K_2")
    if not final_top_k_2_str or not final_top_k_2_str.strip():
        raise ValueError("FINAL_TOP_K_2 is required. Set it in .env file or environment variable (e.g., '10').")
    try:
        final_top_k_2 = int(final_top_k_2_str.strip())
        if final_top_k_2 <= 0:
            raise ValueError(f"final_top_k_2 must be positive, got {final_top_k_2}")
        if final_top_k_2 < final_top_k_1:
            raise ValueError(f"final_top_k_2 ({final_top_k_2}) must be >= final_top_k_1 ({final_top_k_1})")
    except ValueError as e:
        raise ValueError(f"Invalid FINAL_TOP_K_2 value '{final_top_k_2_str}': {e}")
    
    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
    if not reranker_model:
        raise ValueError("RERANKER_MODEL is required. Set it in .env file or environment variable (e.g., 'BAAI/bge-reranker-base').")
    
    reranker_device = os.getenv("RERANKER_DEVICE", "mps").strip() or None
    if reranker_device and reranker_device not in ("mps", "cuda", "cpu"):
        raise ValueError(f"Invalid RERANKER_DEVICE value '{reranker_device}'. Must be 'mps', 'cuda', 'cpu', or empty (auto-detect).")
    
    reranker_batch_size_str = os.getenv("RERANKER_BATCH_SIZE", "32").strip()
    try:
        reranker_batch_size = int(reranker_batch_size_str)
        if reranker_batch_size <= 0:
            raise ValueError(f"reranker_batch_size must be positive, got {reranker_batch_size}")
    except ValueError as e:
        raise ValueError(f"Invalid RERANKER_BATCH_SIZE value '{reranker_batch_size_str}': {e}")
    
    reranker_top_k_str = os.getenv("RERANKER_TOP_K")
    reranker_top_k = None
    if reranker_top_k_str and reranker_top_k_str.strip():
        try:
            reranker_top_k = int(reranker_top_k_str.strip())
            if reranker_top_k <= 0:
                raise ValueError(f"reranker_top_k must be positive, got {reranker_top_k}")
        except ValueError as e:
            raise ValueError(f"Invalid RERANKER_TOP_K value '{reranker_top_k_str}': {e}")
    
    reranker_tokenization_cache_dir_str = os.getenv("RERANKER_TOKENIZATION_CACHE_DIR")
    reranker_tokenization_cache_dir = None
    if reranker_tokenization_cache_dir_str and reranker_tokenization_cache_dir_str.strip():
        reranker_tokenization_cache_dir = REPO_ROOT / reranker_tokenization_cache_dir_str.strip()
    
    reranker_tokenization_cache_enabled_str = os.getenv("RERANKER_TOKENIZATION_CACHE_ENABLED", "True").strip().lower()
    reranker_tokenization_cache_enabled = reranker_tokenization_cache_enabled_str in ("true", "1", "yes")
    
    reranker_max_concurrent_batches_str = os.getenv("RERANKER_MAX_CONCURRENT_BATCHES", "2").strip()
    try:
        reranker_max_concurrent_batches = int(reranker_max_concurrent_batches_str)
        if reranker_max_concurrent_batches <= 0:
            raise ValueError(f"reranker_max_concurrent_batches must be positive, got {reranker_max_concurrent_batches}")
    except ValueError as e:
        raise ValueError(f"Invalid RERANKER_MAX_CONCURRENT_BATCHES value '{reranker_max_concurrent_batches_str}': {e}")
    
    reranker_prefilter_enabled_str = os.getenv("RERANKER_PREFILTER_ENABLED", "True").strip().lower()
    reranker_prefilter_enabled = reranker_prefilter_enabled_str in ("true", "1", "yes")
    
    reranker_prefilter_min_score_str = os.getenv("RERANKER_PREFILTER_MIN_SCORE")
    reranker_prefilter_min_score = None
    if reranker_prefilter_min_score_str and reranker_prefilter_min_score_str.strip():
        try:
            reranker_prefilter_min_score = float(reranker_prefilter_min_score_str.strip())
        except ValueError as e:
            raise ValueError(f"Invalid RERANKER_PREFILTER_MIN_SCORE value '{reranker_prefilter_min_score_str}': {e}")
    
    reranker_prefilter_score_diff_threshold_str = os.getenv("RERANKER_PREFILTER_SCORE_DIFF_THRESHOLD")
    reranker_prefilter_score_diff_threshold = None
    if reranker_prefilter_score_diff_threshold_str and reranker_prefilter_score_diff_threshold_str.strip():
        try:
            reranker_prefilter_score_diff_threshold = float(reranker_prefilter_score_diff_threshold_str.strip())
            if reranker_prefilter_score_diff_threshold <= 0:
                raise ValueError(f"reranker_prefilter_score_diff_threshold must be positive, got {reranker_prefilter_score_diff_threshold}")
        except ValueError as e:
            raise ValueError(f"Invalid RERANKER_PREFILTER_SCORE_DIFF_THRESHOLD value '{reranker_prefilter_score_diff_threshold_str}': {e}")
    
    reranker_prefilter_min_items_str = os.getenv("RERANKER_PREFILTER_MIN_ITEMS", "20").strip()
    try:
        reranker_prefilter_min_items = int(reranker_prefilter_min_items_str)
        if reranker_prefilter_min_items <= 0:
            raise ValueError(f"reranker_prefilter_min_items must be positive, got {reranker_prefilter_min_items}")
    except ValueError as e:
        raise ValueError(f"Invalid RERANKER_PREFILTER_MIN_ITEMS value '{reranker_prefilter_min_items_str}': {e}")
    
    # Initialize hybrid retriever
    print("\nInitializing hybrid retriever...")
    print(f"  • BM25: k1={bm25_k1}, b={bm25_b}")
    if vector_local_model_name:
        print(f"  • Vector: {vector_local_model_name} (local)")
    else:
        print(f"  • Vector: {vector_model_name} (API)")
    print(f"  • Cross-Encoder: {reranker_model} (device={reranker_device or 'auto'}, batch_size={reranker_batch_size})")
    print(f"  • Retrieval: top-{retrieval_top_k} from each method")
    if use_rrf:
        print(f"  • RRF Fusion: enabled (k={rrf_k}, top-{rrf_top_k})")
    else:
        print(f"  • RRF Fusion: disabled (simple merge)")
    print(f"  • Final output: top-{final_top_k_1} and top-{final_top_k_2}")
    
    retriever = HybridRetriever(
        candidates,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        vector_hnsw_index_path=vector_hnsw_index_path,
        retrieval_top_k=retrieval_top_k,
        use_rrf=use_rrf,
        rrf_k=rrf_k,
        rrf_top_k=rrf_top_k,
        final_top_k_1=final_top_k_1,
        final_top_k_2=final_top_k_2,
        reranker_model=reranker_model,
        vector_api_base=vector_api_base,
        vector_api_key=vector_api_key,
        vector_model_name=vector_model_name,
        vector_dimensions=vector_dimensions,
        vector_max_tokens_per_request=vector_max_tokens_per_request,
        vector_max_items_per_batch=vector_max_items_per_batch,
        vector_rpm_limit=vector_rpm_limit,
        vector_timeout=vector_timeout,
        vector_local_model_name=vector_local_model_name,
        vector_normalize_embeddings=vector_normalize_embeddings,
        vector_use_hnsw=vector_use_hnsw,
        vector_hnsw_m=vector_hnsw_m,
        vector_hnsw_ef_construction=vector_hnsw_ef_construction,
        vector_hnsw_ef_search=vector_hnsw_ef_search,
        vector_embeddings_dir=vector_embeddings_dir,
        vector_cache_embeddings=vector_cache_embeddings,
        reranker_device=reranker_device,
        reranker_batch_size=reranker_batch_size,
        reranker_top_k=reranker_top_k,
        reranker_tokenization_cache_dir=reranker_tokenization_cache_dir,
        reranker_tokenization_cache_enabled=reranker_tokenization_cache_enabled,
        reranker_max_concurrent_batches=reranker_max_concurrent_batches,
        reranker_prefilter_enabled=reranker_prefilter_enabled,
        reranker_prefilter_min_score=reranker_prefilter_min_score,
        reranker_prefilter_score_diff_threshold=reranker_prefilter_score_diff_threshold,
        reranker_prefilter_min_items=reranker_prefilter_min_items,
        vector_query_embedding_model=vector_query_embedding_model,
        bm25_cache_dir=bm25_cache_dir,
        bm25_cache_enabled=bm25_cache_enabled,
        bm25_data_source_hash=data_source_hash,
    )
    print("  ✓ Hybrid retriever initialized")
    
    # Run single query and measure timing
    print("\n" + "=" * 100)
    print("Running Query and Measuring Timing...")
    print("=" * 100)
    print(f"\nQuery ID: {query_id}")
    print(f"Query: {query_text}")
    
    result = retriever.search(query_text, query_id=query_id)
    
    # Display query and results
    print("\n" + "=" * 100)
    print("QUERY AND RESULTS")
    print("=" * 100)
    print(f"\nQuery ID: {query_id}")
    print(f"Query: {query_text}")
    
    # Display top results
    print(f"\nTop-{final_top_k_1} Results:")
    print("-" * 100)
    if result.top_5:
        for rank, (candidate, score) in enumerate(result.top_5, 1):
            print(format_item_result(candidate, score, rank))
        scores_5 = [s for _, s in result.top_5]
        if scores_5:
            print(f"\n  Cross-Encoder Score Range: {min(scores_5):.4f} - {max(scores_5):.4f}")
    else:
        print(f"  No top-{final_top_k_1} results found.")
    
    print(f"\nTop-{final_top_k_2} Results:")
    print("-" * 100)
    if result.top_10:
        for rank, (candidate, score) in enumerate(result.top_10, 1):
            print(format_item_result(candidate, score, rank))
        scores_10 = [s for _, s in result.top_10]
        if scores_10:
            print(f"\n  Cross-Encoder Score Range: {min(scores_10):.4f} - {max(scores_10):.4f}")
    else:
        print(f"  No top-{final_top_k_2} results found.")
    
    # Display timing breakdown
    print("\n" + "=" * 100)
    print("TIMING BREAKDOWN")
    print("=" * 100)
    print("\nStage-level Timing:")
    print("-" * 100)
    print(f"  BM25 Retrieval:        {result.bm25_time:8.4f}s")
    print(f"  Vector Retrieval:      {result.vector_time:8.4f}s")
    if use_rrf:
        print(f"  RRF Fusion:            {result.rrf_time:8.4f}s")
    else:
        print(f"  Merge & Deduplicate:   {result.rrf_time:8.4f}s")
    print(f"  Cross-Encoder Re-rank: {result.rerank_time:8.4f}s")
    print("-" * 100)
    print(f"  TOTAL TIME:            {result.total_time:8.4f}s")
    
    # Calculate percentages
    print("\n" + "-" * 100)
    print("Time Distribution (Percentage of Total):")
    print("-" * 100)
    if result.total_time > 0:
        bm25_pct = (result.bm25_time / result.total_time) * 100
        vector_pct = (result.vector_time / result.total_time) * 100
        rrf_pct = (result.rrf_time / result.total_time) * 100
        rerank_pct = (result.rerank_time / result.total_time) * 100
        
        print(f"  BM25 Retrieval:        {bm25_pct:6.2f}%")
        print(f"  Vector Retrieval:      {vector_pct:6.2f}%")
        print(f"  {'RRF Fusion' if use_rrf else 'Merge & Deduplicate':20s}   {rrf_pct:6.2f}%")
        print(f"  Cross-Encoder Re-rank: {rerank_pct:6.2f}%")
    
    # Performance summary
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)
    if result.total_time > 0:
        queries_per_second = 1.0 / result.total_time
        print(f"  Query time: {result.total_time:.4f}s")
        print(f"  Queries per second: {queries_per_second:.2f}")
    
    if result.ce_timing_info:
        total_inference_time = result.ce_timing_info.get('total_inference_time', 0.0)
        num_items = len(result.merged_items) if result.merged_items else 0
        if total_inference_time > 0 and num_items > 0:
            items_per_second = num_items / total_inference_time
            print(f"  Cross-Encoder throughput: {items_per_second:.1f} items/second")
    
    print("\n" + "=" * 100)
    print("Benchmark completed!")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

