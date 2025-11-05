"""Demo script for hybrid search with BM25, Vector, RRF fusion, and Cross-Encoder re-ranking.

This script demonstrates the hybrid retrieval pipeline:
1. BM25 text retrieval (top-50, configurable)
2. Vector retrieval (top-50, configurable)
3. RRF fusion (optional) or merge and deduplicate
4. Cross-Encoder re-ranking (using BAAI/bge-reranker-base)
5. Output top-K results (configurable, e.g., top-5 and top-10)

Usage:
    python scripts/demo_hybrid_search_no_llm.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

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

# Configure logging to show INFO level messages
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


def load_ground_truth(gt_path: Path, query_id: str) -> list[tuple[str, float]]:
    """Load ground truth scores for a specific query.
    
    Returns:
        List of (item_id, score) tuples sorted by score descending.
    """
    ground_truth = []
    if not gt_path.exists():
        return ground_truth
    
    with gt_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Compare as strings to handle type mismatches
            if str(row.get("query_id", "")) == str(query_id):
                item_id = row.get("item_id", "").strip()
                if not item_id:
                    continue
                
                # Handle empty or invalid score values
                score_str = row.get("score", "0.0").strip()
                try:
                    score = float(score_str) if score_str else 0.0
                except (ValueError, TypeError):
                    score = 0.0
                
                ground_truth.append((item_id, score))
    
    # Sort by score descending
    ground_truth.sort(key=lambda x: x[1], reverse=True)
    return ground_truth


def format_ground_truth_item(item_id: str, score: float, rank: int, candidates_dict: dict) -> str:
    """Format a ground truth item for display."""
    candidate = candidates_dict.get(item_id)
    if candidate:
        name = candidate.name[:60] + "..." if len(candidate.name) > 60 else candidate.name
        return f"  {rank:2d}. [{item_id}] {name} (ground truth score: {score:.1f})"
    else:
        return f"  {rank:2d}. [{item_id}] (item not found in candidates) (ground truth score: {score:.1f})"


def main():
    """Demo hybrid search with Cross-Encoder re-ranking."""
    # Load data
    data_dir = REPO_ROOT / "data" / "test"
    items_path = data_dir / "500_items.csv"
    queries_path = data_dir / "10_queries.csv"
    
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    print("=" * 80)
    print("Hybrid Search Demo (Cross-Encoder Re-ranking)")
    print("=" * 80)
    
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
    
    # Get all required parameters from environment variables
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
    # Local model (preferred)
    vector_local_model_name = os.getenv("VECTOR_LOCAL_MODEL_NAME")
    if vector_local_model_name:
        vector_local_model_name = vector_local_model_name.strip()
        if not vector_local_model_name:
            vector_local_model_name = None
    else:
        vector_local_model_name = None
    
    # API parameters (optional, only needed if not using local model)
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
    
    # Validate: must have either local model or API config
    if not vector_local_model_name and not vector_api_key:
        print("\n" + "=" * 80)
        print("ERROR: Missing vector retrieval configuration!")
        print("=" * 80)
        print("Please set one of the following in your .env file:")
        print("\n1. For local model (recommended, no API key needed):")
        print("   VECTOR_LOCAL_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("\n2. For API (requires API key):")
        print("   VECTOR_API_KEY=sk-your-key-here")
        print(f"\n   Expected .env location: {env_path}")
        print("=" * 80)
        return
    
    # Show which mode is being used
    if vector_local_model_name:
        print(f"  ✓ Using local model: {vector_local_model_name}")
        print(f"  ✓ No API key required (using local embeddings)")
    else:
        print(f"  ✓ Using API mode")
        print(f"  ✓ API key loaded (from .env or environment)")
        if vector_api_base:
            print(f"  ✓ API base: {vector_api_base}")
        else:
            print(f"  ✓ API base: https://api.openai.com/v1 (default)")
    
    # API parameters (only used if using API)
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
    
    # HNSW index path - REQUIRED and must exist
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
    
    # Query embedding model (optional, for local query embedding)
    vector_query_embedding_model = os.getenv("VECTOR_QUERY_EMBEDDING_MODEL")
    if vector_query_embedding_model:
        vector_query_embedding_model = vector_query_embedding_model.strip()
        if not vector_query_embedding_model:
            vector_query_embedding_model = None
    
    # Retrieval parameters
    retrieval_top_k_str = os.getenv("RETRIEVAL_TOP_K")
    if not retrieval_top_k_str or not retrieval_top_k_str.strip():
        raise ValueError("RETRIEVAL_TOP_K is required. Set it in .env file or environment variable (e.g., '50').")
    try:
        retrieval_top_k = int(retrieval_top_k_str.strip())
        if retrieval_top_k <= 0:
            raise ValueError(f"retrieval_top_k must be positive, got {retrieval_top_k}")
    except ValueError as e:
        raise ValueError(f"Invalid RETRIEVAL_TOP_K value '{retrieval_top_k_str}': {e}")
    
    # RRF fusion parameters
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
    
    # Final output parameters
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
    
    # Cross-Encoder re-ranking parameters
    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
    if not reranker_model:
        raise ValueError("RERANKER_MODEL is required. Set it in .env file or environment variable (e.g., 'BAAI/bge-reranker-base').")
    
    reranker_device = os.getenv("RERANKER_DEVICE", "mps").strip() or None
    # Validate device: "mps", "cuda", "cpu", or None (auto)
    if reranker_device and reranker_device not in ("mps", "cuda", "cpu"):
        raise ValueError(f"Invalid RERANKER_DEVICE value '{reranker_device}'. Must be 'mps', 'cuda', 'cpu', or empty (auto-detect).")
    
    reranker_batch_size_str = os.getenv("RERANKER_BATCH_SIZE", "32").strip()
    try:
        reranker_batch_size = int(reranker_batch_size_str)
        if reranker_batch_size <= 0:
            raise ValueError(f"reranker_batch_size must be positive, got {reranker_batch_size}")
        if reranker_batch_size not in (32, 64):
            print(f"  ⚠ Warning: RERANKER_BATCH_SIZE={reranker_batch_size} is not 32 or 64. Recommended: 32 or 64 for Mac MPS.")
    except ValueError as e:
        raise ValueError(f"Invalid RERANKER_BATCH_SIZE value '{reranker_batch_size_str}': {e}")
    
    # Reranker top-K parameter (optional, defaults to None = score all items)
    reranker_top_k_str = os.getenv("RERANKER_TOP_K")
    reranker_top_k = None
    if reranker_top_k_str and reranker_top_k_str.strip():
        try:
            reranker_top_k = int(reranker_top_k_str.strip())
            if reranker_top_k <= 0:
                raise ValueError(f"reranker_top_k must be positive, got {reranker_top_k}")
            # Warn if reranker_top_k is smaller than final_top_k_2
            if reranker_top_k < final_top_k_2:
                print(f"  ⚠ Warning: RERANKER_TOP_K={reranker_top_k} is smaller than FINAL_TOP_K_2={final_top_k_2}.")
                print(f"     RERANKER_TOP_K will be automatically adjusted to at least {final_top_k_2}.")
        except ValueError as e:
            raise ValueError(f"Invalid RERANKER_TOP_K value '{reranker_top_k_str}': {e}")
    
    # Pre-tokenization cache parameters
    reranker_tokenization_cache_dir_str = os.getenv("RERANKER_TOKENIZATION_CACHE_DIR")
    reranker_tokenization_cache_dir = None
    if reranker_tokenization_cache_dir_str and reranker_tokenization_cache_dir_str.strip():
        reranker_tokenization_cache_dir = REPO_ROOT / reranker_tokenization_cache_dir_str.strip()
    
    reranker_tokenization_cache_enabled_str = os.getenv("RERANKER_TOKENIZATION_CACHE_ENABLED", "True").strip().lower()
    reranker_tokenization_cache_enabled = reranker_tokenization_cache_enabled_str in ("true", "1", "yes")
    
    # Concurrent batch processing parameter
    reranker_max_concurrent_batches_str = os.getenv("RERANKER_MAX_CONCURRENT_BATCHES", "2").strip()
    try:
        reranker_max_concurrent_batches = int(reranker_max_concurrent_batches_str)
        if reranker_max_concurrent_batches <= 0:
            raise ValueError(f"reranker_max_concurrent_batches must be positive, got {reranker_max_concurrent_batches}")
    except ValueError as e:
        raise ValueError(f"Invalid RERANKER_MAX_CONCURRENT_BATCHES value '{reranker_max_concurrent_batches_str}': {e}")
    
    # Pre-filtering parameters
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
        if vector_query_embedding_model and vector_query_embedding_model != vector_local_model_name:
            print(f"  • Vector: {vector_local_model_name} - Items: Local, Query: {vector_query_embedding_model}")
        else:
            print(f"  • Vector: {vector_local_model_name} - Items & Query: Local")
    else:
        if vector_query_embedding_model:
            print(f"  • Vector: {vector_model_name} (dim={vector_dimensions}) - Items: API, Query: {vector_query_embedding_model}")
        else:
            print(f"  • Vector: {vector_model_name} (dim={vector_dimensions}) - Items & Query: API")
    print(f"  • Cross-Encoder: {reranker_model} (device={reranker_device or 'auto'}, batch_size={reranker_batch_size}, fp16=True)")
    print(f"  • Cross-Encoder optimization: max_concurrent_batches={reranker_max_concurrent_batches}, tokenization_cache={'enabled' if reranker_tokenization_cache_enabled else 'disabled'}")
    if reranker_prefilter_enabled:
        prefilter_info = f"enabled (min_score={reranker_prefilter_min_score}, score_diff_threshold={reranker_prefilter_score_diff_threshold}, min_items={reranker_prefilter_min_items})"
        print(f"  • Pre-filtering: {prefilter_info}")
    else:
        print(f"  • Pre-filtering: disabled")
    if reranker_top_k is not None:
        print(f"  • Cross-Encoder top-K: {reranker_top_k} (will return top-{reranker_top_k} items)")
    else:
        print(f"  • Cross-Encoder top-K: all items (will score all {rrf_top_k} items)")
    print(f"  • Retrieval: top-{retrieval_top_k} from each method")
    if use_rrf:
        print(f"  • RRF Fusion: enabled (k={rrf_k}, top-{rrf_top_k})")
    else:
        print(f"  • RRF Fusion: disabled (simple merge)")
    print(f"  • Final output: top-{final_top_k_1} and top-{final_top_k_2}")
    print(f"  • HNSW Index: {vector_hnsw_index_path.name}")
    
    retriever = HybridRetriever(
        candidates,
        # BM25 parameters (required)
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        # Required parameters (no defaults)
        vector_hnsw_index_path=vector_hnsw_index_path,
        retrieval_top_k=retrieval_top_k,
        use_rrf=use_rrf,
        rrf_k=rrf_k,
        rrf_top_k=rrf_top_k,
        final_top_k_1=final_top_k_1,
        final_top_k_2=final_top_k_2,
        reranker_model=reranker_model,
        # Optional parameters (with defaults)
        # API parameters (optional, only if using API)
        vector_api_base=vector_api_base,
        vector_api_key=vector_api_key,
        vector_model_name=vector_model_name,
        vector_dimensions=vector_dimensions,
        vector_max_tokens_per_request=vector_max_tokens_per_request,
        vector_max_items_per_batch=vector_max_items_per_batch,
        vector_rpm_limit=vector_rpm_limit,
        vector_timeout=vector_timeout,
        # Local model parameters (optional, if using local model)
        vector_local_model_name=vector_local_model_name,
        vector_normalize_embeddings=vector_normalize_embeddings,
        vector_use_hnsw=vector_use_hnsw,
        vector_hnsw_m=vector_hnsw_m,
        vector_hnsw_ef_construction=vector_hnsw_ef_construction,
        vector_hnsw_ef_search=vector_hnsw_ef_search,
        vector_embeddings_dir=vector_embeddings_dir,
        vector_cache_embeddings=vector_cache_embeddings,
        # Cross-Encoder re-ranking optional parameters
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
        # Query embedding parameters (optional)
        vector_query_embedding_model=vector_query_embedding_model,
        # BM25 caching parameters (optional)
        bm25_cache_dir=bm25_cache_dir,
        bm25_cache_enabled=bm25_cache_enabled,
        bm25_data_source_hash=data_source_hash,
    )
    print("  ✓ Hybrid retriever initialized")
    
    # Load a sample query
    print("\nLoading sample query...")
    with queries_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
    
    query_id = first_row.get("id") or first_row.get("query_id") or "demo_query"
    query_text = first_row.get("search_term_pt") or first_row.get("query") or ""
    
    print(f"  Query ID: {query_id}")
    print(f"  Query: {query_text}")
    
    # Load ground truth
    print("\nLoading ground truth scores...")
    gt_path = data_dir / "test_query_new.csv"
    ground_truth = load_ground_truth(gt_path, query_id)
    if ground_truth:
        print(f"  ✓ Loaded {len(ground_truth)} ground truth scores for query {query_id}")
    else:
        print(f"  ⚠ No ground truth scores found for query {query_id} in {gt_path}")
    
    # Create a dictionary for quick candidate lookup
    candidates_dict = {candidate.id: candidate for candidate in candidates}
    
    # Perform hybrid retrieval
    print("\n" + "=" * 80)
    print("Performing Hybrid Search...")
    print("=" * 80)
    
    result = retriever.search(query_text, query_id=query_id)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # 0. Ground Truth (Top-10)
    if ground_truth:
        print("\n" + "-" * 80)
        print("0. Ground Truth Scores (Top-10)")
        print("-" * 80)
        top_10_gt = ground_truth[:10]
        for rank, (item_id, score) in enumerate(top_10_gt, 1):
            print(format_ground_truth_item(item_id, score, rank, candidates_dict))
        if len(ground_truth) > 10:
            print(f"\n  ... (showing top-10 of {len(ground_truth)} ground truth items)")
        print(f"\n  Ground Truth Score Range: {min(s for _, s in top_10_gt):.1f} - {max(s for _, s in top_10_gt):.1f}")
    
    # 1. Merged Items (before Cross-Encoder re-ranking)
    # Note: BM25 and Vector results are not displayed to reduce output clutter
    print("\n" + "-" * 80)
    if use_rrf:
        print(f"1. RRF Fusion Results (Before Cross-Encoder Re-ranking): {len(result.merged_items)} items")
    else:
        print(f"1. Merged Items (Before Cross-Encoder Re-ranking): {len(result.merged_items)} unique items")
    print("-" * 80)
    if use_rrf:
        print(f"  Total items after RRF fusion: {len(result.merged_items)}")
        print(f"  (BM25: {len(result.bm25_results)} + Vector: {len(result.vector_results)} -> {len(result.merged_items)} items via RRF)")
    else:
        print(f"  Total unique items after deduplication: {len(result.merged_items)}")
        print(f"  (BM25: {len(result.bm25_results)} + Vector: {len(result.vector_results)} -> {len(result.merged_items)} unique)")
    
    # 2. Top-K_1 Results (after Cross-Encoder re-ranking)
    print("\n" + "-" * 80)
    print(f"2. Top-{final_top_k_1} Results (After Cross-Encoder Re-ranking)")
    print("-" * 80)
    if result.top_5:
        for rank, (candidate, score) in enumerate(result.top_5, 1):
            print(format_item_result(candidate, score, rank))
        print(f"\n  Cross-Encoder Score Range: {min(s for _, s in result.top_5):.4f} - {max(s for _, s in result.top_5):.4f}")
    else:
        print(f"  No top-{final_top_k_1} results found.")
    
    # 3. Top-K_2 Results (after Cross-Encoder re-ranking)
    print("\n" + "-" * 80)
    print(f"3. Top-{final_top_k_2} Results (After Cross-Encoder Re-ranking)")
    print("-" * 80)
    if result.top_10:
        for rank, (candidate, score) in enumerate(result.top_10, 1):
            print(format_item_result(candidate, score, rank))
        print(f"\n  Cross-Encoder Score Range: {min(s for _, s in result.top_10):.4f} - {max(s for _, s in result.top_10):.4f}")
    else:
        print(f"  No top-{final_top_k_2} results found.")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Query: {query_text}")
    print(f"  Query ID: {query_id}")
    
    if ground_truth:
        print(f"  Ground Truth Items: {len(ground_truth)} items")
        top_gt_ids = {item_id for item_id, _ in ground_truth[:final_top_k_2]}
        if result.top_10:
            top_k2_retrieved_ids = {candidate.id for candidate, _ in result.top_10}
            overlap = len(top_gt_ids & top_k2_retrieved_ids)
            print(f"  Top-{final_top_k_2} Overlap: {overlap}/{final_top_k_2} items match ground truth")
        
        if result.top_5:
            top_k1_retrieved_ids = {candidate.id for candidate, _ in result.top_5}
            overlap_top_k1 = len(top_gt_ids & top_k1_retrieved_ids)
            print(f"  Top-{final_top_k_1} Overlap: {overlap_top_k1}/{final_top_k_1} items match ground truth top-{final_top_k_2}")
    
    print(f"  BM25 Results: {len(result.bm25_results)} items")
    print(f"  Vector Results: {len(result.vector_results)} items")
    print(f"  Merged Items: {len(result.merged_items)} unique items")
    print(f"  Top-{final_top_k_1} Results: {len(result.top_5)} items")
    print(f"  Top-{final_top_k_2} Results: {len(result.top_10)} items")
    
    if result.top_5:
        avg_score_top_k1 = sum(score for _, score in result.top_5) / len(result.top_5)
        print(f"  Average Cross-Encoder Score (Top-{final_top_k_1}): {avg_score_top_k1:.4f}")
    
    # Timing information
    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN")
    print("=" * 80)
    print(f"  BM25 Retrieval:        {result.bm25_time:.3f}s")
    print(f"  Vector Retrieval:      {result.vector_time:.3f}s")
    if use_rrf:
        print(f"  RRF Fusion:            {result.rrf_time:.3f}s")
    else:
        print(f"  Merge & Deduplicate:   {result.rrf_time:.3f}s")
    print(f"  Cross-Encoder Re-rank: {result.rerank_time:.3f}s")
    print("-" * 80)
    print(f"  TOTAL TIME:            {result.total_time:.3f}s")
    print("=" * 80)
    
    # Detailed Cross-Encoder timing breakdown
    if result.ce_timing_info:
        print("\n" + "=" * 80)
        print("CROSS-ENCODER DETAILED TIMING")
        print("=" * 80)
        ce_info = result.ce_timing_info
        print(f"  Build Pairs Time:      {ce_info.get('build_pairs_time', 0):.4f}s")
        print(f"  Total Inference Time:  {ce_info.get('total_inference_time', 0):.4f}s")
        print(f"  Number of Batches:     {ce_info.get('num_batches', 0)}")
        print(f"  Concurrent Processing: {'Yes' if ce_info.get('concurrent', False) else 'No'}")
        
        batch_times = ce_info.get('batch_times', [])
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            max_batch_time = max(batch_times)
            min_batch_time = min(batch_times)
            print(f"  Batch Processing:")
            print(f"    - Average: {avg_batch_time:.4f}s")
            print(f"    - Min:     {min_batch_time:.4f}s")
            print(f"    - Max:     {max_batch_time:.4f}s")
            if len(batch_times) <= 10:
                print(f"    - Individual: {', '.join(f'{t:.4f}s' for t in batch_times)}")
            else:
                print(f"    - Individual (first 5): {', '.join(f'{t:.4f}s' for t in batch_times[:5])} ...")
        
        # Calculate overhead
        total_ce_time = ce_info.get('build_pairs_time', 0) + ce_info.get('total_inference_time', 0)
        overhead = result.rerank_time - total_ce_time
        if overhead > 0.001:
            print(f"  Overhead (sorting, etc.): {overhead:.4f}s ({overhead/result.rerank_time*100:.1f}%)")
        
        # Performance analysis
        print("\n  Performance Analysis:")
        if ce_info.get('total_inference_time', 0) > 0:
            inference_ratio = ce_info.get('total_inference_time', 0) / result.rerank_time
            print(f"    - Inference time ratio: {inference_ratio*100:.1f}% of total CE time")
        
        if batch_times and ce_info.get('concurrent', False):
            # Check if batches are well-balanced
            if max_batch_time > 0:
                imbalance_ratio = (max_batch_time - min_batch_time) / max_batch_time
                if imbalance_ratio > 0.3:
                    print(f"    - ⚠ Batch imbalance detected: {imbalance_ratio*100:.1f}% difference")
                    print(f"      Consider adjusting batch_size or max_concurrent_batches")
                else:
                    print(f"    - ✓ Batches are well-balanced")
        
        # Speedup suggestions
        print("\n  Speedup Suggestions:")
        num_items = len(result.merged_items) if result.merged_items else 0
        
        if ce_info.get('build_pairs_time', 0) > 0.1:
            print(f"    - Build pairs time ({ce_info.get('build_pairs_time', 0):.4f}s) is high")
            print(f"      Consider enabling tokenization cache: RERANKER_TOKENIZATION_CACHE_ENABLED=True")
        
        if ce_info.get('total_inference_time', 0) > 0 and num_items > 0:
            avg_time_per_item = ce_info.get('total_inference_time', 0) / num_items
            total_inference = ce_info.get('total_inference_time', 0)
            
            if avg_time_per_item > 0.1:
                print(f"    - Average time per item ({avg_time_per_item:.4f}s = {avg_time_per_item*1000:.1f}ms) is high")
                
                # Check if using MPS/CUDA
                if reranker_device and reranker_device.lower() in ('mps', 'cuda'):
                    print(f"      ✓ Using {reranker_device.upper()} device (GPU acceleration enabled)")
                    if avg_time_per_item > 0.05:
                        print(f"      ⚠ Despite GPU, still slow. Possible causes:")
                        print(f"        • Model too large (current: {reranker_model})")
                        print(f"        • FP16 not fully utilized")
                        print(f"        • Consider smaller model: BAAI/bge-reranker-v2-m3")
                else:
                    print(f"      ⚠ Not using GPU acceleration (device: {reranker_device or 'auto'})")
                    print(f"      → Set RERANKER_DEVICE=mps (Mac) or RERANKER_DEVICE=cuda (Linux/Windows)")
                
                # Batch size suggestions
                if num_items <= reranker_batch_size:
                    print(f"      • Only {num_items} items (batch_size={reranker_batch_size}), single batch")
                    print(f"        → Increasing batch_size won't help (already fits in one batch)")
                    print(f"        → Consider: reducing pre-filtering to get more items for batch processing")
                else:
                    num_batches = ce_info.get('num_batches', 1)
                    if num_batches > 1:
                        print(f"      • {num_items} items split into {num_batches} batches (batch_size={reranker_batch_size})")
                        if reranker_batch_size < 64:
                            print(f"        → Consider increasing RERANKER_BATCH_SIZE to 64 or 128")
                        if not ce_info.get('concurrent', False):
                            print(f"        → Enable concurrent processing: RERANKER_MAX_CONCURRENT_BATCHES=2 or 4")
                    else:
                        print(f"      • {num_items} items in single batch (batch_size={reranker_batch_size})")
                        if reranker_batch_size < 64 and num_items < 64:
                            print(f"        → Consider increasing RERANKER_BATCH_SIZE to 64 for better GPU utilization")
                
                # Model size suggestions
                if 'large' in reranker_model.lower():
                    print(f"      • Using large model ({reranker_model})")
                    print(f"        → Consider smaller model: BAAI/bge-reranker-v2-m3 (2-3x faster)")
                elif 'v2-m3' in reranker_model.lower():
                    print(f"      • Already using compact model ({reranker_model})")
                    print(f"        → Further speedup: reduce items via pre-filtering or use lighter retrieval")
        
        if not ce_info.get('concurrent', False) and ce_info.get('num_batches', 0) > 1:
            print(f"    - Multiple batches ({ce_info.get('num_batches', 0)}) but concurrent processing disabled")
            print(f"      → Enable: RERANKER_MAX_CONCURRENT_BATCHES=2 or 4")
            print(f"      → Expected speedup: ~{min(2, ce_info.get('num_batches', 0))}x for {ce_info.get('num_batches', 0)} batches")
        
        # Overall performance summary
        if ce_info.get('total_inference_time', 0) > 0 and num_items > 0:
            items_per_second = num_items / ce_info.get('total_inference_time', 1)
            print(f"\n  Performance Summary:")
            print(f"    - Throughput: {items_per_second:.1f} items/second")
            if items_per_second < 10:
                print(f"      ⚠ Low throughput. Target: >20 items/s with GPU, >5 items/s on CPU")
            elif items_per_second < 20:
                print(f"      ⚠ Moderate throughput. Target: >20 items/s with GPU acceleration")
            else:
                print(f"      ✓ Good throughput")
        
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
