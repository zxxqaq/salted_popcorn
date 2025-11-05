# scripts/generate_bm25_cache.py
# !/usr/bin/env python3
"""
Generate BM25 cache for faster cold start.

This script loads candidates from CSV and builds BM25 index,
automatically saving cache to disk for reuse.
"""

import sys
from pathlib import Path

# Add project root to path (required for imports)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bm25_retrieval import load_food_candidates, BM25Retriever


def main():
    # Settings
    items_path = REPO_ROOT / "data" / "test" / "500_items.csv"
    cache_dir = REPO_ROOT / "data" / "bm25_cache"
    k1 = 1.5
    b = 0.75

    if not items_path.exists():
        print(f"❌ Error: Items file not found: {items_path}")
        sys.exit(1)

    print("=" * 80)
    print("BM25 Cache Generator")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Items file: {items_path}")
    print(f"  • Cache directory: {cache_dir}")
    print(f"  • BM25 parameters: k1={k1}, b={b}")
    print("=" * 80 + "\n")

    # Load candidates (auto-cache)
    print("Step 1: Loading candidates...")
    candidates, data_hash = load_food_candidates(
        items_path,
        cache_dir=cache_dir,
        cache_enabled=True,
    )
    print(f"  ✓ Loaded {len(candidates)} candidates\n")

    # Create BM25 retriever (cache index automatically)
    print("Step 2: Building BM25 index...")
    retriever = BM25Retriever(
        candidates,
        k1=k1,
        b=b,
        cache_dir=cache_dir,
        cache_enabled=True,
        data_source_hash=data_hash,
    )

    print("\n" + "=" * 80)
    print("✓ BM25 cache generated successfully!")
    print("=" * 80)
    print(f"\nCache location: {cache_dir}")
    print("You can now use this cache in hybrid retrieval.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cache generation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)