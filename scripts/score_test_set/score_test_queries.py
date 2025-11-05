#!/usr/bin/env python3
"""Score test queries using LLM to evaluate relevance.

This script:
1. Reads queries from data/test/30_queries.csv
2. Reads query-item pairs from data/test/test_query.csv
3. Loads item details from data/raw/5k_items.csv when needed
4. Uses LLM API to score relevance (0-10) for each query-item pair
5. Updates the score column in test_query.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

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

try:
    import openai
except ImportError:
    openai = None

from src.bm25_retrieval import build_candidate_text


def load_queries(queries_path: Path) -> Dict[str, str]:
    """Load queries from CSV file.
    
    Returns:
        Dictionary mapping query_id to query_text.
    """
    queries = {}
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    with queries_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("id") or row.get("query_id") or ""
            query_text = row.get("search_term_pt") or row.get("query") or row.get("text") or ""
            if query_id and query_text.strip():
                queries[query_id] = query_text.strip()
    
    return queries


def load_items_dict(items_path: Path) -> Dict[str, dict]:
    """Load all items from CSV into a dictionary by item_id.
    
    Returns:
        Dictionary mapping item_id to item metadata.
    """
    items_dict = {}
    
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    
    print(f"Loading items from: {items_path}")
    with items_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("itemId") or row.get("item_id") or ""
            if not item_id:
                continue
            
            # Parse itemMetadata JSON
            item_metadata_str = row.get("itemMetadata") or "{}"
            try:
                item_metadata = json.loads(item_metadata_str)
            except json.JSONDecodeError:
                item_metadata = {}
            
            items_dict[item_id] = item_metadata
    
    print(f"  ✓ Loaded {len(items_dict)} items")
    return items_dict


def build_item_text(item_metadata: dict) -> str:
    """Build text representation of item from metadata."""
    return build_candidate_text(item_metadata)


def score_relevance_with_llm(
    query: str,
    item_name: str,
    item_text: str,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> float:
    """Score relevance of item to query using LLM.
    
    Returns:
        Relevance score from 0-10 (float)
    """
    if not openai:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    
    prompt = f"""You are a relevance scoring expert. Please evaluate the relevance of the following item to the query.

Query: {query}

Item Name: {item_name}
Item Details: {item_text}

Scoring Guidelines:
- 10: Perfect match, item perfectly meets the query requirements
- 8-9: Highly relevant, item very well meets the query requirements
- 6-7: Relevant, item basically meets the query requirements
- 5: Moderately relevant, item partially meets the query requirements
- 3-4: Low relevance, item has weak connection to the query
- 1-2: Almost irrelevant, item has minimal connection to the query
- 0: Completely irrelevant, item has no connection to the query

Please return only a number between 0-10, without any other text or explanation."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a relevance scoring expert. Return only a number between 0-10."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Use deterministic temperature for consistent scoring
                max_tokens=10,
            )
            
            score_str = response.choices[0].message.content.strip()
            # Extract numeric score
            import re
            score_match = re.search(r'\d+\.?\d*', score_str)
            if score_match:
                score = float(score_match.group())
                # Clamp to 0-10 range
                score = max(0.0, min(10.0, score))
                return score
            else:
                print(f"  ⚠ Warning: Could not parse score from LLM response: {score_str}")
                return 0.0
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  ⚠ Error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Error after {max_retries} attempts: {e}")
                return 0.0
    
    return 0.0


def main():
    """Score test queries."""
    print("=" * 80)
    print("Score Test Queries with LLM")
    print("=" * 80)
    
    # Load queries
    queries_path = REPO_ROOT / "data" / "test" / "30_queries.csv"
    print(f"\nLoading queries from: {queries_path}")
    queries = load_queries(queries_path)
    if not queries:
        raise ValueError(f"No valid queries found in {queries_path}")
    print(f"  ✓ Loaded {len(queries)} queries")
    
    # Load test query pairs
    test_query_path = REPO_ROOT / "data" / "test" / "test_query.csv"
    if not test_query_path.exists():
        raise FileNotFoundError(f"Test query file not found: {test_query_path}")
    
    print(f"\nLoading test query pairs from: {test_query_path}")
    query_item_pairs = []
    with test_query_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("query_id", "").strip()
            item_id = row.get("item_id", "").strip()
            item_name = row.get("item_name", "").strip()
            existing_score = row.get("score", "").strip()
            
            if query_id and item_id:
                query_item_pairs.append({
                    "query_id": query_id,
                    "item_id": item_id,
                    "item_name": item_name,
                    "existing_score": existing_score,
                })
    
    print(f"  ✓ Loaded {len(query_item_pairs)} query-item pairs")
    
    # Count how many need scoring
    needs_scoring = [p for p in query_item_pairs if not p["existing_score"]]
    print(f"  • {len(needs_scoring)} pairs need scoring")
    print(f"  • {len(query_item_pairs) - len(needs_scoring)} pairs already scored")
    
    if not needs_scoring:
        print("\n✓ All pairs already have scores. Nothing to do.")
        return
    
    # Load items dictionary
    items_path = REPO_ROOT / "data" / "raw" / "5k_items.csv"
    items_dict = load_items_dict(items_path)
    
    # LLM API configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required. Set it in .env file.")
    
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    
    print(f"\nLLM Configuration:")
    print(f"  • Model: {model}")
    print(f"  • API Base: {api_base}")
    
    # Load existing file structure for incremental updates
    print("\n" + "=" * 80)
    print("Scoring query-item pairs (with real-time batch updates)...")
    print("=" * 80)
    
    # Read existing file structure
    all_rows = []
    fieldnames = None
    row_index_map = {}  # Map (query_id, item_id) -> row index
    
    with test_query_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        
        # Ensure 'score' column exists, add if missing
        if "score" not in fieldnames:
            fieldnames.append("score")
            print("  ℹ️  'score' column not found, will be added automatically")
        
        for idx, row in enumerate(reader):
            query_id = row.get("query_id", "").strip()
            item_id = row.get("item_id", "").strip()
            
            # Ensure 'score' field exists in row
            if "score" not in row:
                row["score"] = ""
            
            all_rows.append(row)
            row_index_map[(query_id, item_id)] = idx
    
    # Batch update configuration
    batch_size = 50  # Update file every N scores
    scored_count = 0
    
    def update_file_batch():
        """Update CSV file with current scores."""
        with test_query_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    
    # Score pairs with real-time updates
    for i, pair in enumerate(needs_scoring, 1):
        query_id = pair["query_id"]
        item_id = pair["item_id"]
        item_name = pair["item_name"]
        
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"\n[{i}/{len(needs_scoring)}] ⚠ Query {query_id} not found, skipping item {item_id}")
            continue
        
        # Get item metadata
        item_metadata = items_dict.get(item_id)
        if not item_metadata:
            print(f"\n[{i}/{len(needs_scoring)}] ⚠ Item {item_id} not found in items file, using name only")
            item_text = item_name
        else:
            item_text = build_item_text(item_metadata)
        
        print(f"\n[{i}/{len(needs_scoring)}] Query {query_id}: {query_text[:50]}{'...' if len(query_text) > 50 else ''}")
        print(f"  Item: {item_name[:60]}{'...' if len(item_name) > 60 else ''}")
        
        # Score with LLM
        score = score_relevance_with_llm(
            query=query_text,
            item_name=item_name,
            item_text=item_text,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )
        
        print(f"  Score: {score:.1f}/10")
        
        # Update score in memory
        row_idx = row_index_map.get((query_id, item_id))
        if row_idx is not None:
            all_rows[row_idx]["score"] = f"{score:.1f}"
            scored_count += 1
        
        # Batch update: save to file every N scores
        if scored_count % batch_size == 0:
            update_file_batch()
            print(f"  💾 Saved progress: {scored_count}/{len(needs_scoring)} scores updated")
        
        # Rate limiting - small delay to avoid API limits
        time.sleep(0.5)
    
    # Final update: save all remaining scores
    print("\n" + "=" * 80)
    print("Final update: saving all scores to file...")
    print("=" * 80)
    update_file_batch()
    print(f"  ✓ Updated {scored_count} scores")
    
    # Statistics
    total_scored_count = sum(1 for row in all_rows if row.get("score") and row["score"].strip())
    print(f"\nStatistics:")
    print(f"  • Total pairs: {len(all_rows)}")
    print(f"  • Pairs with scores: {total_scored_count}")
    print(f"  • Pairs without scores: {len(all_rows) - total_scored_count}")
    
    if total_scored_count > 0:
        scores = [float(row["score"]) for row in all_rows if row.get("score") and row["score"].strip()]
        if scores:
            avg_score = sum(scores) / len(scores)
            high_relevance = sum(1 for s in scores if s >= 5.0)
            low_relevance = sum(1 for s in scores if s < 5.0)
            print(f"  • Average score: {avg_score:.2f}/10")
            print(f"  • High relevance (≥5): {high_relevance} ({high_relevance/len(scores)*100:.1f}%)")
            print(f"  • Low relevance (<5): {low_relevance} ({low_relevance/len(scores)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Scoring completed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Scoring cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

