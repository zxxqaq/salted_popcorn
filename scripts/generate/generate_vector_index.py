#!/usr/bin/env python3
"""
Generate HNSW vector index for local SentenceTransformer model.

This script:
1. Loads candidates from CSV file
2. Generates embeddings using local SentenceTransformer model
3. Builds HNSW index
4. Saves index and embeddings cache to disk

Usage:
    python scripts/generate_vector_index.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]  # scripts/generate/ -> scripts/ -> project root
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.bm25_retrieval import load_food_candidates
from src.vector_retrieval import VectorRetriever

# Load environment variables from .env file
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded .env file from: {env_path}")
else:
    print(f"⚠ No .env file found at {env_path}")
    print("  Using environment variables only...")

def main():
    print("=" * 80)
    print("HNSW Vector Index Generator")
    print("=" * 80)
    print("\nThis script will generate HNSW index and embeddings cache for local model.")
    print("=" * 80 + "\n")
    
    # Get configuration from environment variables
    print("1. Loading configuration from environment variables...")
    
    # Local model name (required)
    local_model_name = os.getenv("VECTOR_LOCAL_MODEL_NAME")
    if not local_model_name or not local_model_name.strip():
        raise ValueError(
            "VECTOR_LOCAL_MODEL_NAME is required. Set it in .env file.\n"
            "Example: VECTOR_LOCAL_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    local_model_name = local_model_name.strip()
    print(f"   ✓ Local model: {local_model_name}")
    
    # Data file path
    items_path_str = os.getenv("ITEMS_PATH", "data/raw/5k_items.csv")
    items_path = project_root / items_path_str
    if not items_path.exists():
        raise ValueError(
            f"Items file not found: {items_path}\n"
            f"Set ITEMS_PATH in .env file or use default: data/raw/5k_items.csv"
        )
    print(f"   ✓ Items file: {items_path}")
    
    # Index path (directory or file)
    index_path_str = os.getenv("VECTOR_HNSW_INDEX_PATH")
    if not index_path_str or not index_path_str.strip():
        # Default: use data/vector_indices directory, auto-generate filename
        index_path = project_root / "data" / "vector_indices"
    else:
        index_path = project_root / index_path_str.strip()
        # If it's a file path, use it as-is
        # If it's a directory path, auto-generate filename
        if index_path.suffix == ".index":
            # It's a file path
            pass
        else:
            # It's a directory path
            pass
    
    print(f"   ✓ Index path: {index_path}")
    
    # HNSW parameters
    use_hnsw_str = os.getenv("VECTOR_USE_HNSW", "True").strip().lower()
    use_hnsw = use_hnsw_str in ("true", "1", "yes")
    
    hnsw_m_str = os.getenv("VECTOR_HNSW_M", "32").strip()
    try:
        hnsw_m = int(hnsw_m_str)
    except ValueError:
        hnsw_m = 32
    
    hnsw_ef_construction_str = os.getenv("VECTOR_HNSW_EF_CONSTRUCTION", "100").strip()
    try:
        hnsw_ef_construction = int(hnsw_ef_construction_str)
    except ValueError:
        hnsw_ef_construction = 100
    
    hnsw_ef_search_str = os.getenv("VECTOR_HNSW_EF_SEARCH", "64").strip()
    try:
        hnsw_ef_search = int(hnsw_ef_search_str)
    except ValueError:
        hnsw_ef_search = 64
    
    print(f"   ✓ HNSW parameters: M={hnsw_m}, ef_construction={hnsw_ef_construction}, ef_search={hnsw_ef_search}")
    
    # Embeddings caching
    cache_embeddings_str = os.getenv("VECTOR_CACHE_EMBEDDINGS", "True").strip().lower()
    cache_embeddings = cache_embeddings_str in ("true", "1", "yes")
    
    embeddings_dir_str = os.getenv("VECTOR_EMBEDDINGS_DIR")
    embeddings_dir = None
    if embeddings_dir_str and embeddings_dir_str.strip():
        embeddings_dir = project_root / embeddings_dir_str.strip()
    else:
        # Default: same as index directory
        if index_path.suffix == ".index":
            embeddings_dir = index_path.parent
        else:
            embeddings_dir = index_path
    
    print(f"   ✓ Embeddings cache: {'enabled' if cache_embeddings else 'disabled'}")
    if cache_embeddings and embeddings_dir:
        print(f"   ✓ Embeddings directory: {embeddings_dir}")
    
    # Normalize embeddings
    normalize_embeddings_str = os.getenv("VECTOR_NORMALIZE_EMBEDDINGS", "True").strip().lower()
    normalize_embeddings = normalize_embeddings_str in ("true", "1", "yes")
    print(f"   ✓ Normalize embeddings: {normalize_embeddings}")
    
    print("\n2. Loading candidates...")
    candidates, data_source_hash = load_food_candidates(items_path)
    print(f"   ✓ Loaded {len(candidates)} candidates")
    
    print("\n3. Generating embeddings and building HNSW index...")
    print("   This may take a few minutes depending on the number of items...")
    print("-" * 80)
    
    # Initialize VectorRetriever with local model
    # This will automatically:
    # 1. Generate embeddings using local model
    # 2. Build HNSW index
    # 3. Save index and embeddings cache
    retriever = VectorRetriever(
        candidates,
        local_model_name=local_model_name,
        normalize_embeddings=normalize_embeddings,
        use_hnsw=use_hnsw,
        index_path=index_path,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        embeddings_dir=embeddings_dir,
        cache_embeddings=cache_embeddings,
    )
    
    print("-" * 80)
    print("\n4. Index generation complete!")
    print("=" * 80)
    
    # Get actual index path (either explicit or generated)
    if retriever.index_path is not None:
        actual_index_path = retriever.index_path
    else:
        actual_index_path = retriever._get_index_path(
            retriever.model_name, retriever.dimensions, len(candidates)
        )
    
    if actual_index_path and actual_index_path.exists():
        index_size_mb = actual_index_path.stat().st_size / (1024 * 1024)
        print(f"  • Index file: {actual_index_path}")
        print(f"  • Index size: {index_size_mb:.2f} MB")
        print(f"  • Index dimension: {retriever.hnsw_index.dim if retriever.hnsw_index else 'N/A'}")
        print(f"  • Number of vectors: {retriever.hnsw_index.element_count if retriever.hnsw_index else 'N/A'}")
    
    # Show embeddings cache info
    if cache_embeddings and retriever.embeddings_dir:
        cache_path = retriever._get_embeddings_cache_path(
            retriever.model_name, retriever.dimensions, len(candidates)
        )
        if cache_path and cache_path.exists():
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"  • Embeddings cache: {cache_path.name}")
            print(f"  • Cache size: {cache_size_mb:.2f} MB")
    
    print(f"  • Model: {retriever.model_name}")  # Use actual model_name from retriever
    print(f"  • Dimension: {retriever.dimensions}")
    print(f"  • Normalize: {normalize_embeddings}")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print(f"1. Update your .env file with the index path:")
    print(f"   VECTOR_HNSW_INDEX_PATH={actual_index_path.relative_to(project_root) if actual_index_path else 'N/A'}")
    print("\n2. You can now use this index in hybrid_retrieval_no_llm.py")
    print("   The index will be automatically loaded on subsequent runs.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Index generation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

