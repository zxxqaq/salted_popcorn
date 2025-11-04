"""Demo script for hybrid search with BM25, Vector, and LLM re-ranking.

This script demonstrates the hybrid retrieval pipeline:
1. BM25 text retrieval (top-50, configurable)
2. Vector retrieval (top-50, configurable)
3. Merge and deduplicate
4. LLM re-ranking
5. Output top-K results (configurable, e.g., top-5 and top-10)

Usage:
    python scripts/demo_hybrid_search.py
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

from src.hybrid_retrieval import HybridRetriever, load_food_candidates_for_hybrid


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
    """Demo hybrid search with detailed output."""
    # Load data
    data_dir = REPO_ROOT / "data" / "test"
    items_path = data_dir / "500_items.csv"
    queries_path = data_dir / "10_queries.csv"
    
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    print("=" * 80)
    print("Hybrid Search Demo")
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
    candidates = load_food_candidates_for_hybrid(items_path)
    print(f"  ✓ Loaded {len(candidates)} items")
    
    # Get API keys from environment (loaded from .env file or system environment)
    llm_api_key = os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        print("\n" + "=" * 80)
        print("ERROR: OPENAI_API_KEY not found!")
        print("=" * 80)
        print("Please set your OpenAI API key in one of the following ways:")
        print("\n1. Create a .env file in the project root with:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        print(f"\n   Expected location: {env_path}")
        print("\n2. Or set environment variable:")
        print("   export OPENAI_API_KEY='sk-your-key-here'")
        print("\n3. Install python-dotenv if you want to use .env file:")
        print("   pip install python-dotenv")
        print("=" * 80)
        return
    
    llm_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    if llm_api_base.strip() == "":
        llm_api_base = "https://api.openai.com/v1"
    
    print(f"  ✓ LLM API key loaded (from .env or environment)")
    print(f"  ✓ LLM API base: {llm_api_base}")
    
    # Get all required parameters from environment variables
    print("\nLoading configuration from environment variables...")
    
    # LLM parameters
    llm_model = os.getenv("LLM_MODEL")
    if not llm_model or not llm_model.strip():
        raise ValueError("LLM_MODEL is required. Set it in .env file or environment variable (e.g., 'gpt-4o-mini').")
    llm_model = llm_model.strip()
    
    llm_max_tokens_per_item_str = os.getenv("LLM_MAX_TOKENS_PER_ITEM")
    if not llm_max_tokens_per_item_str or not llm_max_tokens_per_item_str.strip():
        raise ValueError("LLM_MAX_TOKENS_PER_ITEM is required. Set it in .env file or environment variable (e.g., '200').")
    try:
        llm_max_tokens_per_item = int(llm_max_tokens_per_item_str.strip())
        if llm_max_tokens_per_item <= 0:
            raise ValueError(f"llm_max_tokens_per_item must be positive, got {llm_max_tokens_per_item}")
    except ValueError as e:
        raise ValueError(f"Invalid LLM_MAX_TOKENS_PER_ITEM value '{llm_max_tokens_per_item_str}': {e}")
    
    llm_max_context_tokens_str = os.getenv("LLM_MAX_CONTEXT_TOKENS")
    if not llm_max_context_tokens_str or not llm_max_context_tokens_str.strip():
        raise ValueError("LLM_MAX_CONTEXT_TOKENS is required. Set it in .env file or environment variable (e.g., '128000').")
    try:
        llm_max_context_tokens = int(llm_max_context_tokens_str.strip())
        if llm_max_context_tokens <= 0:
            raise ValueError(f"llm_max_context_tokens must be positive, got {llm_max_context_tokens}")
    except ValueError as e:
        raise ValueError(f"Invalid LLM_MAX_CONTEXT_TOKENS value '{llm_max_context_tokens_str}': {e}")
    
    llm_reserved_output_tokens_str = os.getenv("LLM_RESERVED_OUTPUT_TOKENS")
    if not llm_reserved_output_tokens_str or not llm_reserved_output_tokens_str.strip():
        raise ValueError("LLM_RESERVED_OUTPUT_TOKENS is required. Set it in .env file or environment variable (e.g., '8000').")
    try:
        llm_reserved_output_tokens = int(llm_reserved_output_tokens_str.strip())
        if llm_reserved_output_tokens <= 0:
            raise ValueError(f"llm_reserved_output_tokens must be positive, got {llm_reserved_output_tokens}")
    except ValueError as e:
        raise ValueError(f"Invalid LLM_RESERVED_OUTPUT_TOKENS value '{llm_reserved_output_tokens_str}': {e}")
    
    llm_tokens_per_item_output_str = os.getenv("LLM_TOKENS_PER_ITEM_OUTPUT")
    if not llm_tokens_per_item_output_str or not llm_tokens_per_item_output_str.strip():
        raise ValueError("LLM_TOKENS_PER_ITEM_OUTPUT is required. Set it in .env file or environment variable (e.g., '60').")
    try:
        llm_tokens_per_item_output = int(llm_tokens_per_item_output_str.strip())
        if llm_tokens_per_item_output <= 0:
            raise ValueError(f"llm_tokens_per_item_output must be positive, got {llm_tokens_per_item_output}")
    except ValueError as e:
        raise ValueError(f"Invalid LLM_TOKENS_PER_ITEM_OUTPUT value '{llm_tokens_per_item_output_str}': {e}")
    
    llm_sleep_str = os.getenv("LLM_SLEEP", "0.0").strip()
    try:
        llm_sleep = float(llm_sleep_str)
        if llm_sleep < 0:
            raise ValueError(f"llm_sleep must be non-negative, got {llm_sleep}")
    except ValueError as e:
        raise ValueError(f"Invalid LLM_SLEEP value '{llm_sleep_str}': {e}")
    
    llm_timeout_str = os.getenv("LLM_TIMEOUT")
    if not llm_timeout_str or not llm_timeout_str.strip():
        raise ValueError("LLM_TIMEOUT is required. Set it in .env file or environment variable (e.g., '120.0').")
    try:
        llm_timeout = float(llm_timeout_str.strip())
        if llm_timeout <= 0:
            raise ValueError(f"llm_timeout must be positive, got {llm_timeout}")
    except ValueError as e:
        raise ValueError(f"Invalid LLM_TIMEOUT value '{llm_timeout_str}': {e}")
    
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
    
    # Vector retrieval parameters
    vector_api_base = os.getenv("VECTOR_API_BASE")
    if not vector_api_base or not vector_api_base.strip():
        raise ValueError("VECTOR_API_BASE is required. Set it in .env file or environment variable (e.g., 'https://api.openai.com/v1').")
    vector_api_base = vector_api_base.strip()
    
    vector_api_key = os.getenv("VECTOR_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not vector_api_key:
        raise ValueError("VECTOR_API_KEY or OPENAI_API_KEY is required. Set it in .env file or environment variable.")
    
    vector_model_name = os.getenv("VECTOR_MODEL_NAME")
    if not vector_model_name or not vector_model_name.strip():
        raise ValueError("VECTOR_MODEL_NAME is required. Set it in .env file or environment variable (e.g., 'text-embedding-3-small').")
    vector_model_name = vector_model_name.strip()
    
    vector_dimensions_str = os.getenv("VECTOR_DIMENSIONS")
    if not vector_dimensions_str or not vector_dimensions_str.strip():
        raise ValueError("VECTOR_DIMENSIONS is required. Set it in .env file or environment variable (e.g., '1536').")
    try:
        vector_dimensions = int(vector_dimensions_str.strip())
        if vector_dimensions <= 0:
            raise ValueError(f"vector_dimensions must be positive, got {vector_dimensions}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_DIMENSIONS value '{vector_dimensions_str}': {e}")
    
    vector_max_tokens_per_request_str = os.getenv("VECTOR_MAX_TOKENS_PER_REQUEST")
    if not vector_max_tokens_per_request_str or not vector_max_tokens_per_request_str.strip():
        raise ValueError("VECTOR_MAX_TOKENS_PER_REQUEST is required. Set it in .env file or environment variable (e.g., '8192').")
    try:
        vector_max_tokens_per_request = int(vector_max_tokens_per_request_str.strip())
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
    
    vector_rpm_limit_str = os.getenv("VECTOR_RPM_LIMIT")
    if not vector_rpm_limit_str or not vector_rpm_limit_str.strip():
        raise ValueError("VECTOR_RPM_LIMIT is required. Set it in .env file or environment variable (e.g., '300').")
    try:
        vector_rpm_limit = int(vector_rpm_limit_str.strip())
        if vector_rpm_limit <= 0:
            raise ValueError(f"vector_rpm_limit must be positive, got {vector_rpm_limit}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_RPM_LIMIT value '{vector_rpm_limit_str}': {e}")
    
    vector_timeout_str = os.getenv("VECTOR_TIMEOUT")
    if not vector_timeout_str or not vector_timeout_str.strip():
        raise ValueError("VECTOR_TIMEOUT is required. Set it in .env file or environment variable (e.g., '120.0').")
    try:
        vector_timeout = float(vector_timeout_str.strip())
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
    
    # Initialize hybrid retriever
    print("\nInitializing hybrid retriever...")
    print(f"  • BM25: k1={bm25_k1}, b={bm25_b}")
    print(f"  • Vector: {vector_model_name} (dim={vector_dimensions})")
    print(f"  • LLM: {llm_model} for re-ranking")
    print(f"  • Retrieval: top-{retrieval_top_k} from each method")
    if use_rrf:
        print(f"  • RRF Fusion: enabled (k={rrf_k}, top-{rrf_top_k})")
    else:
        print(f"  • RRF Fusion: disabled (simple merge)")
    print(f"  • Final output: top-{final_top_k_1} and top-{final_top_k_2}")
    print(f"  • HNSW Index: {vector_hnsw_index_path.name}")
    
    retriever = HybridRetriever(
        candidates,
        # LLM re-ranking parameters (required)
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_max_tokens_per_item=llm_max_tokens_per_item,
        llm_max_context_tokens=llm_max_context_tokens,
        llm_reserved_output_tokens=llm_reserved_output_tokens,
        llm_tokens_per_item_output=llm_tokens_per_item_output,
        llm_sleep=llm_sleep,
        llm_timeout=llm_timeout,
        # BM25 parameters (required)
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        # Vector retrieval parameters (required)
        vector_api_base=vector_api_base,
        vector_api_key=vector_api_key,
        vector_model_name=vector_model_name,
        vector_dimensions=vector_dimensions,
        vector_max_tokens_per_request=vector_max_tokens_per_request,
        vector_max_items_per_batch=vector_max_items_per_batch,
        vector_rpm_limit=vector_rpm_limit,
        vector_timeout=vector_timeout,
        vector_normalize_embeddings=vector_normalize_embeddings,
        vector_hnsw_index_path=vector_hnsw_index_path,
        vector_use_hnsw=vector_use_hnsw,
        vector_hnsw_m=vector_hnsw_m,
        vector_hnsw_ef_construction=vector_hnsw_ef_construction,
        vector_hnsw_ef_search=vector_hnsw_ef_search,
        vector_embeddings_dir=vector_embeddings_dir,
        vector_cache_embeddings=vector_cache_embeddings,
        # Retrieval parameters (required)
        retrieval_top_k=retrieval_top_k,
        # RRF fusion parameters (required)
        use_rrf=use_rrf,
        rrf_k=rrf_k,
        rrf_top_k=rrf_top_k,
        # Final output parameters (required)
        final_top_k_1=final_top_k_1,
        final_top_k_2=final_top_k_2,
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
    
    # 1. BM25 Results (top-50)
    print("\n" + "-" * 80)
    print(f"1. BM25 Text Retrieval Results (Top-{retrieval_top_k})")
    print("-" * 80)
    if result.bm25_results:
        display_count = min(retrieval_top_k, len(result.bm25_results))
        for rank, (candidate, score) in enumerate(result.bm25_results[:display_count], 1):
            print(format_item_result(candidate, score, rank))
        if len(result.bm25_results) > retrieval_top_k:
            print(f"\n  ... (showing top-{retrieval_top_k} of {len(result.bm25_results)} results)")
    else:
        print("  No BM25 results found.")
    
    # 2. Vector Results (top-50)
    print("\n" + "-" * 80)
    print(f"2. Vector Retrieval Results (Top-{retrieval_top_k})")
    print("-" * 80)
    if result.vector_results:
        display_count = min(retrieval_top_k, len(result.vector_results))
        for rank, (candidate, score) in enumerate(result.vector_results[:display_count], 1):
            print(format_item_result(candidate, score, rank))
        if len(result.vector_results) > retrieval_top_k:
            print(f"\n  ... (showing top-{retrieval_top_k} of {len(result.vector_results)} results)")
    else:
        print("  No vector results found.")
    
    # 3. Merged Items (before LLM re-ranking)
    print("\n" + "-" * 80)
    if use_rrf:
        print(f"3. RRF Fusion Results (Before LLM Re-ranking): {len(result.merged_items)} items")
    else:
        print(f"3. Merged Items (Before LLM Re-ranking): {len(result.merged_items)} unique items")
    print("-" * 80)
    if use_rrf:
        print(f"  Total items after RRF fusion: {len(result.merged_items)}")
        print(f"  (BM25: {len(result.bm25_results)} + Vector: {len(result.vector_results)} -> {len(result.merged_items)} items via RRF)")
    else:
        print(f"  Total unique items after deduplication: {len(result.merged_items)}")
        print(f"  (BM25: {len(result.bm25_results)} + Vector: {len(result.vector_results)} -> {len(result.merged_items)} unique)")
    
    # 4. Top-K_1 Results (after LLM re-ranking)
    print("\n" + "-" * 80)
    print(f"4. Top-{final_top_k_1} Results (After LLM Re-ranking)")
    print("-" * 80)
    if result.top_5:
        for rank, (candidate, score) in enumerate(result.top_5, 1):
            print(format_item_result(candidate, score, rank))
        print(f"\n  LLM Score Range: {min(s for _, s in result.top_5):.1f} - {max(s for _, s in result.top_5):.1f}")
    else:
        print(f"  No top-{final_top_k_1} results found.")
    
    # 5. Top-K_2 Results (after LLM re-ranking)
    print("\n" + "-" * 80)
    print(f"5. Top-{final_top_k_2} Results (After LLM Re-ranking)")
    print("-" * 80)
    if result.top_10:
        for rank, (candidate, score) in enumerate(result.top_10, 1):
            print(format_item_result(candidate, score, rank))
        print(f"\n  LLM Score Range: {min(s for _, s in result.top_10):.1f} - {max(s for _, s in result.top_10):.1f}")
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
        print(f"  Average LLM Score (Top-{final_top_k_1}): {avg_score_top_k1:.2f}")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

