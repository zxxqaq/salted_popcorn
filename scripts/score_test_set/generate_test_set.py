#!/usr/bin/env python3
"""Generate test set from queries using BM25 and Vector retrieval.

This script:
1. Reads queries from data/test/30_queries.csv
2. Performs BM25 retrieval (top-50) and Vector retrieval (top-50)
3. Merges and deduplicates results
4. Saves results to data/test/test_query.csv in format: query_id, item_id, item_name
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Set

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

from src.bm25_retrieval import BM25Retriever, load_food_candidates, Candidates
from src.vector_retrieval import VectorRetriever


def load_queries(queries_path: Path) -> list[tuple[str, str]]:
    """Load queries from CSV file.
    
    Returns:
        List of (query_id, query_text) tuples.
    """
    queries = []
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    with queries_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("id") or row.get("query_id") or ""
            query_text = row.get("search_term_pt") or row.get("query") or row.get("text") or ""
            if query_id and query_text.strip():
                queries.append((query_id, query_text.strip()))
    
    return queries


def merge_and_deduplicate(
    bm25_results: list[tuple[Candidates, float]],
    vector_results: list[tuple[Candidates, float]]
) -> list[Candidates]:
    """Merge BM25 and Vector results and deduplicate by item ID.
    
    Returns:
        List of unique Candidates, preserving order (BM25 first, then Vector).
    """
    seen_ids: Set[str] = set()
    merged: list[Candidates] = []
    
    # Add BM25 results first
    for candidate, _ in bm25_results:
        if candidate.id not in seen_ids:
            seen_ids.add(candidate.id)
            merged.append(candidate)
    
    # Add Vector results (skip duplicates)
    for candidate, _ in vector_results:
        if candidate.id not in seen_ids:
            seen_ids.add(candidate.id)
            merged.append(candidate)
    
    return merged


def main():
    """Generate test set from queries."""
    print("=" * 80)
    print("Generate Test Set")
    print("=" * 80)
    
    # Load queries
    queries_path = REPO_ROOT / "data" / "test" / "30_queries.csv"
    print(f"\nLoading queries from: {queries_path}")
    queries = load_queries(queries_path)
    if not queries:
        raise ValueError(f"No valid queries found in {queries_path}")
    print(f"  ✓ Loaded {len(queries)} queries")
    
    # Load items
    items_path_str = os.getenv("ITEMS_PATH", "data/raw/5k_items.csv")
    items_path = REPO_ROOT / items_path_str.strip()
    
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    
    print(f"\nLoading items from: {items_path}")
    
    # BM25 caching parameters
    bm25_cache_dir_str = os.getenv("BM25_CACHE_DIR")
    bm25_cache_dir = None
    if bm25_cache_dir_str and bm25_cache_dir_str.strip():
        bm25_cache_dir = REPO_ROOT / bm25_cache_dir_str.strip()
    
    bm25_cache_enabled_str = os.getenv("BM25_CACHE_ENABLED", "True").strip().lower()
    bm25_cache_enabled = bm25_cache_enabled_str in ("true", "1", "yes")
    
    candidates, data_source_hash = load_food_candidates(
        items_path,
        cache_dir=bm25_cache_dir,
        cache_enabled=bm25_cache_enabled,
    )
    print(f"  ✓ Loaded {len(candidates)} items")
    
    # BM25 parameters
    print("\nLoading configuration from environment variables...")
    
    bm25_k1_str = os.getenv("BM25_K1")
    if not bm25_k1_str or not bm25_k1_str.strip():
        raise ValueError("BM25_K1 is required. Set it in .env file.")
    try:
        bm25_k1 = float(bm25_k1_str.strip())
        if bm25_k1 <= 0:
            raise ValueError(f"bm25_k1 must be positive, got {bm25_k1}")
    except ValueError as e:
        raise ValueError(f"Invalid BM25_K1 value '{bm25_k1_str}': {e}")
    
    bm25_b_str = os.getenv("BM25_B")
    if not bm25_b_str or not bm25_b_str.strip():
        raise ValueError("BM25_B is required. Set it in .env file.")
    try:
        bm25_b = float(bm25_b_str.strip())
        if bm25_b < 0 or bm25_b > 1:
            raise ValueError(f"bm25_b must be between 0 and 1, got {bm25_b}")
    except ValueError as e:
        raise ValueError(f"Invalid BM25_B value '{bm25_b_str}': {e}")
    
    # Vector retrieval parameters
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
        raise ValueError(
            "VECTOR_LOCAL_MODEL_NAME or VECTOR_API_KEY is required. "
            "Set one of them in .env file."
        )
    
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
        raise ValueError("VECTOR_HNSW_INDEX_PATH is required. Set it in .env file.")
    vector_hnsw_index_path = REPO_ROOT / vector_hnsw_index_path_str.strip()
    
    if not vector_hnsw_index_path.exists():
        raise ValueError(f"HNSW index file does not exist: {vector_hnsw_index_path}")
    
    vector_use_hnsw_str = os.getenv("VECTOR_USE_HNSW", "True").strip().lower()
    vector_use_hnsw = vector_use_hnsw_str in ("true", "1", "yes")
    
    vector_hnsw_m_str = os.getenv("VECTOR_HNSW_M")
    if not vector_hnsw_m_str or not vector_hnsw_m_str.strip():
        raise ValueError("VECTOR_HNSW_M is required. Set it in .env file.")
    try:
        vector_hnsw_m = int(vector_hnsw_m_str.strip())
        if vector_hnsw_m <= 0:
            raise ValueError(f"vector_hnsw_m must be positive, got {vector_hnsw_m}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_HNSW_M value '{vector_hnsw_m_str}': {e}")
    
    vector_hnsw_ef_construction_str = os.getenv("VECTOR_HNSW_EF_CONSTRUCTION")
    if not vector_hnsw_ef_construction_str or not vector_hnsw_ef_construction_str.strip():
        raise ValueError("VECTOR_HNSW_EF_CONSTRUCTION is required. Set it in .env file.")
    try:
        vector_hnsw_ef_construction = int(vector_hnsw_ef_construction_str.strip())
        if vector_hnsw_ef_construction <= 0:
            raise ValueError(f"vector_hnsw_ef_construction must be positive, got {vector_hnsw_ef_construction}")
    except ValueError as e:
        raise ValueError(f"Invalid VECTOR_HNSW_EF_CONSTRUCTION value '{vector_hnsw_ef_construction_str}': {e}")
    
    vector_hnsw_ef_search_str = os.getenv("VECTOR_HNSW_EF_SEARCH")
    if not vector_hnsw_ef_search_str or not vector_hnsw_ef_search_str.strip():
        raise ValueError("VECTOR_HNSW_EF_SEARCH is required. Set it in .env file.")
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
    
    # Initialize retrievers
    print("\nInitializing retrievers...")
    print(f"  • BM25: k1={bm25_k1}, b={bm25_b}")
    if vector_local_model_name:
        print(f"  • Vector: {vector_local_model_name} (local)")
    else:
        print(f"  • Vector: {vector_model_name} (API)")
    
    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever(
        candidates,
        k1=bm25_k1,
        b=bm25_b,
        cache_dir=bm25_cache_dir,
        cache_enabled=bm25_cache_enabled,
        data_source_hash=data_source_hash,
    )
    print("  ✓ BM25 retriever initialized")
    
    # Initialize Vector retriever
    vector_retriever = VectorRetriever(
        candidates,
        # API parameters (optional, if using API)
        api_base=vector_api_base,
        api_key=vector_api_key,
        model_name=vector_model_name,
        dimensions=vector_dimensions,
        max_tokens_per_request=vector_max_tokens_per_request,
        max_items_per_batch=vector_max_items_per_batch,
        rpm_limit=vector_rpm_limit,
        timeout=vector_timeout,
        # Local model parameters (optional, if using local model)
        local_model_name=vector_local_model_name,
        normalize_embeddings=vector_normalize_embeddings,
        # HNSW index parameters
        use_hnsw=vector_use_hnsw,
        index_path=vector_hnsw_index_path,
        hnsw_m=vector_hnsw_m,
        hnsw_ef_construction=vector_hnsw_ef_construction,
        hnsw_ef_search=vector_hnsw_ef_search,
        embeddings_dir=vector_embeddings_dir,
        cache_embeddings=vector_cache_embeddings,
        # Query embedding parameters
        query_embedding_model=vector_query_embedding_model,
    )
    print("  ✓ Vector retriever initialized")
    
    # Process queries
    print("\n" + "=" * 80)
    print("Processing queries...")
    print("=" * 80)
    
    top_k = 50  # Top-50 from each method
    all_results = []
    
    for i, (query_id, query_text) in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query {query_id}: {query_text[:60]}{'...' if len(query_text) > 60 else ''}")
        
        # BM25 retrieval
        bm25_results = bm25_retriever.search(query_text, top_k=top_k)
        print(f"  BM25: {len(bm25_results)} results")
        
        # Vector retrieval
        vector_results = vector_retriever.search(query_text, top_k=top_k)
        print(f"  Vector: {len(vector_results)} results")
        
        # Merge and deduplicate
        merged_items = merge_and_deduplicate(bm25_results, vector_results)
        print(f"  Merged: {len(merged_items)} unique items")
        
        # Store results
        for candidate in merged_items:
            all_results.append((query_id, candidate.id, candidate.name))
    
    # Save results
    output_path = REPO_ROOT / "data" / "test" / "test_query.csv"
    print(f"\n" + "=" * 80)
    print(f"Saving results to: {output_path}")
    print("=" * 80)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'item_id', 'item_name'])
        for query_id, item_id, item_name in all_results:
            writer.writerow([query_id, item_id, item_name])
    
    print(f"  ✓ Saved {len(all_results)} results")
    print(f"  ✓ {len(queries)} queries processed")
    
    # Statistics
    queries_with_results = len(set(query_id for query_id, _, _ in all_results))
    avg_items_per_query = len(all_results) / len(queries) if queries else 0
    
    print(f"\nStatistics:")
    print(f"  • Queries processed: {len(queries)}")
    print(f"  • Queries with results: {queries_with_results}")
    print(f"  • Total item-query pairs: {len(all_results)}")
    print(f"  • Average items per query: {avg_items_per_query:.1f}")
    
    print("\n" + "=" * 80)
    print("Test set generation completed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

