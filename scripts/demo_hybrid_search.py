"""Demo script for hybrid search with BM25, Vector, and LLM re-ranking.

This script demonstrates the hybrid retrieval pipeline:
1. BM25 text retrieval (top-20)
2. Vector retrieval (top-20)
3. Merge and deduplicate
4. LLM re-ranking
5. Output top-5 and top-10 results

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
    
    # Initialize hybrid retriever
    print("\nInitializing hybrid retriever...")
    print("  • BM25: default parameters (k1=1.5, b=0.75)")
    print("  • Vector: local SentenceTransformer model")
    print("  • LLM: OpenAI API for re-ranking")
    print("  • Retrieval: top-20 from each method")
    
    retriever = HybridRetriever(
        candidates,
        # LLM re-ranking (required parameters)
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        # BM25 parameters
        bm25_k1=1.5,
        bm25_b=0.75,
        # Vector retrieval (local model)
        vector_model_name="text-embedding-3-large",
        vector_api_base=llm_api_base,  # None = use local model
        vector_api_key=llm_api_key,
        vector_dimensions=768,
        # LLM re-ranking (optional parameters)
        llm_model="gpt-4o-mini",
        llm_max_tokens_per_item=200,  # Max tokens per item text (truncate if exceeded)
        llm_sleep=0.0,
        # Retrieval parameters
        retrieval_top_k=20,  # Top-20 from each method
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
    
    # 1. BM25 Results (top-20)
    print("\n" + "-" * 80)
    print("1. BM25 Text Retrieval Results (Top-20)")
    print("-" * 80)
    if result.bm25_results:
        display_count = min(20, len(result.bm25_results))
        for rank, (candidate, score) in enumerate(result.bm25_results[:display_count], 1):
            print(format_item_result(candidate, score, rank))
        if len(result.bm25_results) > 20:
            print(f"\n  ... (showing top-20 of {len(result.bm25_results)} results)")
    else:
        print("  No BM25 results found.")
    
    # 2. Vector Results (top-20)
    print("\n" + "-" * 80)
    print("2. Vector Retrieval Results (Top-20)")
    print("-" * 80)
    if result.vector_results:
        display_count = min(20, len(result.vector_results))
        for rank, (candidate, score) in enumerate(result.vector_results[:display_count], 1):
            print(format_item_result(candidate, score, rank))
        if len(result.vector_results) > 20:
            print(f"\n  ... (showing top-20 of {len(result.vector_results)} results)")
    else:
        print("  No vector results found.")
    
    # 3. Merged Items (before LLM re-ranking)
    print("\n" + "-" * 80)
    print(f"3. Merged Items (Before LLM Re-ranking): {len(result.merged_items)} unique items")
    print("-" * 80)
    print(f"  Total unique items after deduplication: {len(result.merged_items)}")
    print(f"  (BM25: {len(result.bm25_results)} + Vector: {len(result.vector_results)} -> {len(result.merged_items)} unique)")
    
    # 4. Top-5 Results (after LLM re-ranking)
    print("\n" + "-" * 80)
    print("4. Top-5 Results (After LLM Re-ranking)")
    print("-" * 80)
    if result.top_5:
        for rank, (candidate, score) in enumerate(result.top_5, 1):
            print(format_item_result(candidate, score, rank))
        print(f"\n  LLM Score Range: {min(s for _, s in result.top_5):.1f} - {max(s for _, s in result.top_5):.1f}")
    else:
        print("  No top-5 results found.")
    
    # 5. Top-10 Results (after LLM re-ranking)
    print("\n" + "-" * 80)
    print("5. Top-10 Results (After LLM Re-ranking)")
    print("-" * 80)
    if result.top_10:
        for rank, (candidate, score) in enumerate(result.top_10, 1):
            print(format_item_result(candidate, score, rank))
        print(f"\n  LLM Score Range: {min(s for _, s in result.top_10):.1f} - {max(s for _, s in result.top_10):.1f}")
    else:
        print("  No top-10 results found.")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Query: {query_text}")
    print(f"  Query ID: {query_id}")
    
    if ground_truth:
        print(f"  Ground Truth Items: {len(ground_truth)} items")
        top_10_gt_ids = {item_id for item_id, _ in ground_truth[:10]}
        if result.top_10:
            top_10_retrieved_ids = {candidate.id for candidate, _ in result.top_10}
            overlap = len(top_10_gt_ids & top_10_retrieved_ids)
            print(f"  Top-10 Overlap: {overlap}/10 items match ground truth")
        
        if result.top_5:
            top_5_retrieved_ids = {candidate.id for candidate, _ in result.top_5}
            overlap_top5 = len(top_10_gt_ids & top_5_retrieved_ids)
            print(f"  Top-5 Overlap: {overlap_top5}/5 items match ground truth top-10")
    
    print(f"  BM25 Results: {len(result.bm25_results)} items")
    print(f"  Vector Results: {len(result.vector_results)} items")
    print(f"  Merged Items: {len(result.merged_items)} unique items")
    print(f"  Top-5 Results: {len(result.top_5)} items")
    print(f"  Top-10 Results: {len(result.top_10)} items")
    
    if result.top_5:
        avg_score_top5 = sum(score for _, score in result.top_5) / len(result.top_5)
        print(f"  Average LLM Score (Top-5): {avg_score_top5:.2f}")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

