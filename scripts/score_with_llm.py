"""Generate query-item relevance scores using a LiteLLM-compatible API.

This script iterates over all queries and items, requests an LLM to rate
the relevance on a 0-10 integer scale, and stores the results in a CSV
(`query_id,item_id,score`).

Example usage:
    python scripts/score_with_llm.py \
        --queries data/test/10_queries.csv \
        --items data/test/500_items.csv \
        --output data/test/test_query.csv \
        --model gpt-4.1-mini \
        --api-base https://api.openai.com \
        --api-key YOUR_API_KEY \
        --rpm-limit 500 \
        --batch-size 30

The script streams results directly to disk to avoid high memory usage.
Be aware that scoring 100 x 5,000 pairs requires a very large number of
LLM calls. Consider batching multiple items per request and monitoring
your API quota/costs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import requests

# Ensure repository root is on sys.path for imports when executed as a script.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bm25_retrieval import build_candidate_text


LOGGER = logging.getLogger("score_with_llm")


def read_queries(path: Path) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                "id": row.get("id") or row.get("query_id") or str(len(queries) + 1),
                "text": row.get("search_term_pt") or row.get("query") or "",
            })
    return queries


def read_items(path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("itemMetadata")
            if not raw:
                continue
            metadata = json.loads(raw)
            text = build_candidate_text(metadata)
            items.append({
                "id": row.get("itemId", ""),
                "name": metadata.get("name", ""),
                "text": text,
            })
    return items


def chunked(seq: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def build_prompt(query: str, batch: Sequence[Dict[str, str]], max_text_length: int = 500) -> str:
    lines = [
        "You are a ranking assistant for food search.",
        "",
        "Score each candidate item for the user query on an integer scale 0-10.",
        "",
        "Scoring guidelines:",
        "- 10: Perfectly matches the query",
        "- 8-9: Highly relevant, clear connection",
        "- 6-7: Somewhat relevant, partial match",
        "- 4-5: Marginally relevant, weak connection",
        "- 2-3: Barely relevant, minimal connection",
        "- 0-1: Completely unrelated",
        "",
        "Important:",
        "- Use the full range of scores. Don't be too strict.",
        "- Even items with weak connections should get scores 2-5, not 0.",
        "- Only give 0 if the item is completely unrelated to food or the query.",
        "- Consider using intermediate scores (3-7) for items that are somewhat relevant.",
        "",
        "Respond ONLY with a JSON array of objects: [{\"item_id\": ..., \"score\": ...}, ...]",
        "",
        f"Query: {query}",
        "",
        "Candidates:",
    ]
    for idx, item in enumerate(batch, start=1):
        snippet = item["text"][:max_text_length]
        lines.append(f"{idx}. item_id={item['id']} | text={snippet}")
    lines.append("")
    lines.append("Return JSON now:")
    prompt = "\n".join(lines)
    return prompt


def call_llm(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    batch_size: int = 10,
    timeout: float = 120.0,
) -> str:
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Dynamically calculate max_tokens based on batch_size
    # Each item's JSON response: item_id (UUID 36 chars) + score + JSON format ≈ 80-100 tokens
    # Plus array format overhead
    estimated_response_tokens = batch_size * 100 + 100
    max_tokens = max(1000, estimated_response_tokens)  # At least 1000 tokens
    
    LOGGER.debug(
        "Request: batch_size=%d, prompt_length=%d chars, max_tokens=%d",
        batch_size, len(prompt), max_tokens
    )
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    LOGGER.debug("POST %s", url)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"LLM request failed ({response.status_code}): {response.text}")
    data = response.json()
    message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # Log token usage
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    
    # Check if max_tokens limit is reached (response may be truncated)
    if completion_tokens >= max_tokens * 0.95:  # Reaching 95% indicates possible truncation
        LOGGER.warning(
            "Response may be truncated: completion_tokens=%d, max_tokens=%d",
            completion_tokens, max_tokens
        )
    
    LOGGER.debug(
        "LLM response: prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
        prompt_tokens, completion_tokens, total_tokens
    )
    
    return message


def parse_scores(raw_text: str) -> List[Dict[str, float]]:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model response as JSON: {text}") from exc


def score_batch(
    query: Dict[str, str],
    batch: Sequence[Dict[str, str]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    sleep: float,
    max_text_length: int = 500,
) -> List[Dict[str, float]]:
    prompt = build_prompt(query["text"], batch, max_text_length=max_text_length)
    raw = call_llm(
        api_base=api_base,
        api_key=api_key,
        model=model,
        prompt=prompt,
        batch_size=len(batch),
    )
    scores = parse_scores(raw)
    if sleep > 0:
        time.sleep(sleep)
    return scores


def run_scoring(args: argparse.Namespace) -> None:
    queries = read_queries(args.queries)
    items = read_items(args.items)
    LOGGER.info("Loaded %d queries and %d items", len(queries), len(items))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["query_id", "item_id", "score"])
        writer.writeheader()

        for q_idx, query in enumerate(queries, start=1):
            LOGGER.info("Scoring query %s (%d/%d)", query["id"], q_idx, len(queries))
            for batch in chunked(items, args.batch_size):
                try:
                    scored = score_batch(
                        query,
                        batch,
                        api_base=args.api_base,
                        api_key=args.api_key,
                        model=args.model,
                        sleep=args.sleep,
                        max_text_length=args.max_text_length,
                    )
                except Exception as exc:
                    LOGGER.error("Failed to score query %s batch starting with %s: %s", query["id"], batch[0]["id"], exc)
                    continue

                # Build score_map and record returned item_ids
                score_map = {}
                returned_item_ids = []
                for entry in scored:
                    item_id = entry.get("item_id")
                    if item_id:
                        returned_item_ids.append(item_id)
                        score_map[item_id] = float(entry.get("score", 0))
                
                LOGGER.debug(
                    "Batch size: %d, LLM returned scores for %d items",
                    len(batch), len(score_map)
                )
                
                # Check for missing item_ids
                missing_items = [item["id"] for item in batch if item["id"] not in score_map]
                if missing_items:
                    # Try case-insensitive matching (in case LLM returns inconsistent format)
                    missing_found = []
                    for missing_id in missing_items[:]:
                        for returned_id in returned_item_ids:
                            if missing_id.lower() == returned_id.lower():
                                score_map[missing_id] = score_map.pop(returned_id)
                                missing_items.remove(missing_id)
                                missing_found.append(f"{returned_id}→{missing_id}")
                                break
                    
                    if missing_found:
                        LOGGER.info("Fixed %d item_id case mismatches: %s", len(missing_found), ", ".join(missing_found[:3]))
                    
                    if missing_items:
                        LOGGER.warning(
                            "Missing scores for %d items in batch. Expected: %d, Got: %d. Missing IDs: %s",
                            len(missing_items),
                            len(batch),
                            len(score_map),
                            ", ".join(missing_items[:5]) if len(missing_items) > 5 else ", ".join(missing_items)
                        )
                        # Display all item_ids returned by LLM (for debugging)
                        LOGGER.debug(
                            "LLM returned item_ids: %s",
                            ", ".join(returned_item_ids[:10]) if len(returned_item_ids) > 10 else ", ".join(returned_item_ids)
                        )
                        LOGGER.debug(
                            "Expected item_ids: %s",
                            ", ".join([item["id"] for item in batch][:10]) if len(batch) > 10 else ", ".join([item["id"] for item in batch])
                        )
                
                for item in batch:
                    score_value = score_map.get(item["id"])
                    writer.writerow({
                        "query_id": query["id"],
                        "item_id": item["id"],
                        "score": score_value if score_value is not None else "",
                    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate relevance scores with LLM")
    parser.add_argument("--queries", type=Path, required=True, help="Path to queries.csv")
    parser.add_argument("--items", type=Path, required=True, help="Path to items CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--api-base", dest="api_base", required=True, help="LiteLLM proxy base URL")
    parser.add_argument("--api-key", dest="api_key", required=True, help="API key for the proxy")
    parser.add_argument("--model", required=True, help="Model name to use (e.g. text-embedding-3-small)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of items per LLM request")
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=500,
        help="Maximum text length per item in prompt (default: 500). Reduce if prompt is too long.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between requests (seconds). If --rpm-limit is set, this will be auto-calculated.",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=0,
        help="Requests per minute limit. If set, automatically calculates --sleep to stay within limit (default: 0 = disabled). For gpt-4.1-mini, recommended: 500",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...)")
    args = parser.parse_args()
    
    # Initialize logging early so we can use LOGGER in parse_args logic
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    
    # Auto-calculate sleep time based on RPM limit
    if args.rpm_limit > 0:
        # Ensure we stay under the limit: sleep = (60 seconds / rpm_limit) - estimated_request_time
        # Assume each request takes ~2 seconds on average
        estimated_request_time = 2.0
        calculated_sleep = max(0.0, (60.0 / args.rpm_limit) - estimated_request_time)
        
        # If RPM limit is very high (>100), sleep time should be very small or 0
        # For RPM=500, sleep should be around 0.1 seconds or less
        if calculated_sleep < 0.01:
            calculated_sleep = 0.0
            LOGGER.info(
                "RPM limit %d is very high. No sleep delay needed (rate limit will not be reached).",
                args.rpm_limit
            )
        
        if args.sleep > 0 and args.sleep != calculated_sleep:
            LOGGER.warning(
                "Both --sleep (%.1f) and --rpm-limit (%d) are set. "
                "Using calculated sleep time %.1f based on RPM limit.",
                args.sleep, args.rpm_limit, calculated_sleep
            )
        args.sleep = calculated_sleep
        if calculated_sleep > 0:
            LOGGER.info(
                "RPM limit set to %d. Auto-calculated sleep time: %.2f seconds per request",
                args.rpm_limit, args.sleep
            )
    
    return args


def main() -> None:
    args = parse_args()
    # Logging is already initialized in parse_args if rpm_limit is used
    if not logging.getLogger().handlers:
        logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_scoring(args)


if __name__ == "__main__":
    main()

