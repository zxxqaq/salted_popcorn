#!/usr/bin/env python3
"""
Generate pre-tokenization cache for reranker.

This script loads candidates from CSV and pre-tokenizes all item texts,
saving the cache to disk for faster reranking operations.
"""

import os
import sys
from pathlib import Path

# Add project root to path (required for imports)
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

from src.bm25_retrieval import load_food_candidates
from src.reranker import LightweightReranker


def main():
    # Settings - read from environment variables or use defaults
    items_path_str = os.getenv("ITEMS_PATH", "data/test/500_items.csv")
    items_path = REPO_ROOT / items_path_str.strip()

    cache_dir_str = os.getenv("RERANKER_TOKENIZATION_CACHE_DIR")
    if cache_dir_str and cache_dir_str.strip():
        cache_dir = REPO_ROOT / cache_dir_str.strip()
    else:
        cache_dir = REPO_ROOT / "artifacts" / "reranker_tokenization_cache"

    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
    reranker_device = os.getenv("RERANKER_DEVICE", "mps").strip() or None

    if not items_path.exists():
        print(f"❌ Error: Items file not found: {items_path}")
        sys.exit(1)

    print("=" * 80)
    print("Reranker Pre-tokenization Cache Generator")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Items file: {items_path}")
    print(f"  • Cache directory: {cache_dir}")
    print(f"  • Reranker model: {reranker_model}")
    print(f"  • Device: {reranker_device or 'auto'}")
    print("=" * 80 + "\n")

    # Load candidates
    print("Step 1: Loading candidates...")
    candidates, _ = load_food_candidates(items_path)
    print(f"  ✓ Loaded {len(candidates)} candidates\n")

    # Initialize reranker (this will initialize the tokenizer)
    print("Step 2: Initializing reranker and tokenizer...")
    try:
        reranker = LightweightReranker(
            model_name=reranker_model,
            device=reranker_device,
            batch_size=32,  # Not used for tokenization, but required
            tokenization_cache_dir=cache_dir,
            tokenization_cache_enabled=True,
        )
        print(f"  ✓ Reranker initialized (tokenizer ready)\n")
    except Exception as e:
        print(f"  ❌ Failed to initialize reranker: {e}")
        sys.exit(1)

    # Pre-tokenize all items
    print("Step 3: Pre-tokenizing all items...")
    items = [(c.id, c.text) for c in candidates]

    total_items = len(items)
    tokenized_count = 0
    cached_count = 0

    # Process in batches for progress reporting
    batch_size = 100
    for i in range(0, total_items, batch_size):
        batch_items = items[i:i + batch_size]

        # Pre-tokenize this batch (this will cache them)
        for item_id, item_text in batch_items:
            if item_id in reranker.tokenized_items_cache:
                cached_count += 1
            else:
                # Tokenize and cache
                try:
                    if reranker.tokenizer:
                        tokens = reranker.tokenizer.encode(
                            item_text,
                            add_special_tokens=False,
                            return_attention_mask=False,
                            return_tensors=None,
                        )
                        reranker.tokenized_items_cache[item_id] = tokens
                        tokenized_count += 1
                    else:
                        print(f"  ⚠ Warning: Tokenizer not available, skipping item {item_id}")
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to tokenize item {item_id}: {e}")

        # Progress update
        processed = min(i + batch_size, total_items)
        print(f"  Progress: {processed}/{total_items} items ({processed * 100 // total_items}%)")

        # Save cache periodically
        if (i + batch_size) % 500 == 0 or processed == total_items:
            reranker.save_cache()

    # Final save
    print("\nStep 4: Saving final cache...")
    reranker.save_cache()

    # Report statistics
    cache_path = reranker._get_cache_path()
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024) if cache_path.exists() else 0

    print("\n" + "=" * 80)
    print("✓ Pre-tokenization cache generated successfully!")
    print("=" * 80)
    print(f"\nStatistics:")
    print(f"  • Total items: {total_items}")
    print(f"  • Newly tokenized: {tokenized_count}")
    print(f"  • Already cached: {cached_count}")
    print(f"  • Cache file: {cache_path}")
    print(f"  • Cache size: {cache_size_mb:.2f} MB")
    print("\nYou can now use this cache in hybrid retrieval for faster reranking.")
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