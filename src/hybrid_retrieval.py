"""Hybrid retrieval module combining BM25 and Vector retrieval with LLM re-ranking.

This module implements a two-stage retrieval and re-ranking pipeline:
1. BM25 retrieval (top-20)
2. Vector retrieval (top-20)
3. Merge and deduplicate results
4. LLM re-scoring and re-ranking
5. Return top-5 and top-10 results
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import requests

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Ensure repository root is on sys.path for module imports
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bm25_retrieval import BM25Retriever, Candidates, build_candidate_text
from src.vector_retrieval import VectorRetriever

LOGGER = logging.getLogger("hybrid_retrieval")


@dataclass
class HybridRetrievalResult:
    """Result from hybrid retrieval."""
    query_id: str
    query_text: str
    top_5: List[Tuple[Candidates, float]]  # (candidate, llm_score)
    top_10: List[Tuple[Candidates, float]]  # (candidate, llm_score)
    bm25_results: List[Tuple[Candidates, float]]  # Original BM25 results (top-20)
    vector_results: List[Tuple[Candidates, float]]  # Original vector results (top-20)
    merged_items: List[Candidates]  # Merged and deduplicated items before LLM scoring


class HybridRetriever:
    """
    Hybrid retriever that combines BM25 and Vector retrieval with LLM re-ranking.
    """
    
    def __init__(
        self,
        candidates: Sequence[Candidates],
        # LLM re-ranking parameters (required, must come before optional params)
        llm_api_base: str,
        llm_api_key: str,
        # BM25 parameters
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        # Vector retrieval parameters
        vector_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        vector_api_base: str | None = None,
        vector_api_key: str | None = None,
        vector_dimensions: int | None = None,
        # LLM re-ranking parameters (optional)
        llm_model: str = "gpt-4o-mini",
        llm_max_tokens_per_item: int = 200,  # Max tokens per item text (truncate if exceeded)
        llm_max_context_tokens: int = 128000,  # Model context window (128k for gpt-4o/gpt-4-turbo)
        llm_reserved_output_tokens: int = 8000,  # Reserve tokens for response (conservative estimate, increased for large batches)
        llm_tokens_per_item_output: int = 60,  # Estimated tokens per item in output JSON (~50-70 tokens)
        llm_sleep: float = 0.0,
        llm_timeout: float = 120.0,
        # Retrieval parameters
        retrieval_top_k: int = 20,  # Top-K for each retrieval method
    ) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            candidates: Sequence of candidate documents
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            vector_model_name: Vector model name (for local) or API model name
            vector_api_base: Vector API base URL (None for local model)
            vector_api_key: Vector API key (required if vector_api_base is set)
            vector_dimensions: Vector embedding dimensions (for API models)
            llm_api_base: LLM API base URL
            llm_api_key: LLM API key
            llm_model: LLM model name for re-ranking
            llm_max_tokens_per_item: Maximum tokens allowed per item text (truncated if exceeded)
            llm_max_context_tokens: Maximum context window tokens for the model (default: 128k for gpt-4o)
            llm_reserved_output_tokens: Base reserved tokens for LLM response output (will be adjusted per batch)
            llm_tokens_per_item_output: Estimated tokens per item in output JSON (~50-70 tokens per item)
            llm_sleep: Sleep time between LLM requests
            llm_timeout: LLM request timeout
            retrieval_top_k: Top-K results to retrieve from each method (default: 20)
        """
        self.candidates = list(candidates)
        self.retrieval_top_k = retrieval_top_k
        
        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever(
            candidates,
            k1=bm25_k1,
            b=bm25_b,
        )
        
        # Initialize Vector retriever
        self.vector_retriever = VectorRetriever(
            candidates,
            model_name=vector_model_name,
            api_base=vector_api_base,
            api_key=vector_api_key,
            dimensions=vector_dimensions,
        )
        
        # LLM parameters
        self.llm_api_base = llm_api_base.rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_max_tokens_per_item = llm_max_tokens_per_item
        self.llm_max_context_tokens = llm_max_context_tokens
        self.llm_reserved_output_tokens = llm_reserved_output_tokens
        self.llm_tokens_per_item_output = llm_tokens_per_item_output
        self.llm_sleep = llm_sleep
        self.llm_timeout = llm_timeout
    
    def _merge_and_deduplicate(
        self,
        bm25_results: List[Tuple[Candidates, float]],
        vector_results: List[Tuple[Candidates, float]],
    ) -> List[Candidates]:
        """
        Merge results from BM25 and Vector retrieval, removing duplicates.
        
        Args:
            bm25_results: BM25 retrieval results
            vector_results: Vector retrieval results
            
        Returns:
            List of unique candidates (preserving order: BM25 first, then Vector)
        """
        seen_ids = set()
        merged = []
        
        # Add BM25 results first (preserving order)
        for candidate, _ in bm25_results:
            if candidate.id not in seen_ids:
                seen_ids.add(candidate.id)
                merged.append(candidate)
        
        # Add Vector results (preserving order, skipping duplicates)
        for candidate, _ in vector_results:
            if candidate.id not in seen_ids:
                seen_ids.add(candidate.id)
                merged.append(candidate)
        
        LOGGER.debug(
            "Merged %d BM25 + %d Vector results -> %d unique items",
            len(bm25_results), len(vector_results), len(merged)
        )
        
        return merged
    
    def _build_rerank_prompt(
        self,
        query: str,
        items: Sequence[Candidates],
    ) -> str:
        """
        Build prompt for LLM re-ranking.
        
        Args:
            query: User query
            items: Items to re-rank
            
        Returns:
            Prompt string
        """
        # Truncate item texts if needed (based on tokens, not characters)
        item_texts = []
        for item in items:
            text = self._truncate_text_to_tokens(item.text, self.llm_max_tokens_per_item)
            item_texts.append(text)
        
        if items:
            items_list = "\n".join(
                f"{i+1}. item_id: {item.id}, name: {item.name}, text: {text}"
                for i, (item, text) in enumerate(zip(items, item_texts))
            )
        else:
            items_list = "(no items)"
        
        prompt = f"""You are a ranking assistant for food search.

Score each candidate item for the user query on an integer scale 0-10.

Query: {query}

Candidates:
{items_list}

Scoring guidelines:
- 10: Perfectly matches the query
- 8-9: Highly relevant, clear connection
- 6-7: Somewhat relevant, partial match
- 4-5: Marginally relevant, weak connection
- 2-3: Barely relevant, minimal connection
- 0-1: Completely unrelated

Important:
- Use the full range of scores. Don't be too strict.
- Consider intermediate scores (2-7) for items with partial relevance.
- Only assign 0 for completely unrelated items.
- Assign higher scores to items that clearly match the query intent.

Return a JSON array of objects, each with "item_id" and "score" (integer 0-10).
Example format:
[
  {{"item_id": "item_001", "score": 8}},
  {{"item_id": "item_002", "score": 6}},
  ...
]

Return ONLY the JSON array, no other text."""
        
        return prompt
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        if tiktoken is None:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
        
        # Use cl100k_base encoding (used by GPT-4 models)
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback if encoding fails
            return len(text) // 4
    
    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if tiktoken is None:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."
        
        # Use cl100k_base encoding (used by GPT-4 models)
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)
            return truncated_text
        except Exception:
            # Fallback if encoding fails
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."
    
    def _batch_items_by_tokens(
        self,
        query: str,
        items: Sequence[Candidates],
    ) -> List[List[Candidates]]:
        """
        Batch items based on token limits to avoid exceeding context window.
        
        Args:
            query: User query text
            items: Items to batch
            
        Returns:
            List of batches, each batch is a list of Candidates
        """
        # Calculate base prompt tokens (template without items)
        base_prompt = self._build_rerank_prompt(query, [])
        base_tokens = self._estimate_tokens(base_prompt)
        
        # Estimate output tokens: base reserve + tokens per item
        # For batching, we use a conservative estimate (assuming max items per batch)
        # The actual output tokens will be calculated per batch in _call_llm_for_single_batch
        estimated_output_tokens = self.llm_reserved_output_tokens
        
        # Available tokens for items = total context - base prompt - estimated output
        available_tokens = self.llm_max_context_tokens - base_tokens - estimated_output_tokens
        
        if available_tokens <= 0:
            LOGGER.warning(
                "Base prompt tokens (%d) + estimated output tokens (%d) exceed context window (%d). "
                "Using single item per batch.",
                base_tokens, estimated_output_tokens, self.llm_max_context_tokens
            )
            return [[item] for item in items]
        
        batches: List[List[Candidates]] = []
        current_batch: List[Candidates] = []
        current_batch_tokens = 0
        
        for item in items:
            # Truncate item text if needed (based on tokens, not characters)
            text = self._truncate_text_to_tokens(item.text, self.llm_max_tokens_per_item)
            
            # Estimate tokens for this item in the prompt format
            item_line = f"{len(current_batch)+1}. item_id: {item.id}, name: {item.name}, text: {text}"
            item_tokens = self._estimate_tokens(item_line)
            
            # Check if adding this item would exceed the limit
            if current_batch_tokens + item_tokens > available_tokens and current_batch:
                # Start a new batch
                batches.append(current_batch)
                current_batch = [item]
                current_batch_tokens = item_tokens
            else:
                # Add to current batch
                current_batch.append(item)
                current_batch_tokens += item_tokens
        
        # Add remaining batch
        if current_batch:
            batches.append(current_batch)
        
        LOGGER.debug(
            "Batched %d items into %d batches (available_tokens=%d per batch)",
            len(items), len(batches), available_tokens
        )
        
        return batches
    
    def _call_llm_for_single_batch(
        self,
        query: str,
        items: Sequence[Candidates],
    ) -> List[Dict[str, float]]:
        """
        Call LLM to re-score a single batch of items.
        
        Args:
            query: User query
            items: Items to re-score (single batch)
            
        Returns:
            List of dicts with "item_id" and "score"
        """
        if not items:
            return []
        
        prompt = self._build_rerank_prompt(query, items)
        
        # Calculate max_tokens based on number of items
        # Each item output: {"item_id": "...", "score": X} ≈ 50-70 tokens
        # Add buffer for JSON array format and potential variations
        estimated_response_tokens = len(items) * self.llm_tokens_per_item_output + 200  # Buffer for JSON array format
        max_tokens = max(1000, estimated_response_tokens)
        
        # Ensure we don't exceed context window (but allow up to reasonable limit)
        # The actual prompt + output should not exceed context window
        prompt_tokens = self._estimate_tokens(prompt)
        max_output_tokens = self.llm_max_context_tokens - prompt_tokens - 1000  # Safety margin
        max_tokens = min(max_tokens, max_output_tokens)
        
        if max_tokens < 100:
            LOGGER.warning(
                "Estimated max_tokens (%d) is too small for %d items. Using minimum 1000.",
                max_tokens, len(items)
            )
            max_tokens = 1000
        
        # Handle URL construction: api_base might already include /v1
        api_base_clean = self.llm_api_base.rstrip("/")
        if api_base_clean.endswith("/v1"):
            url = api_base_clean + "/chat/completions"
        else:
            url = api_base_clean + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        
        prompt_tokens = self._estimate_tokens(prompt)
        LOGGER.debug(
            "LLM batch request: %d items, prompt_tokens≈%d, max_tokens=%d",
            len(items), prompt_tokens, max_tokens
        )
        
        # Log request details for debugging
        LOGGER.debug("LLM API URL: %s", url)
        LOGGER.debug("LLM API Base: %s", self.llm_api_base)
        
        response = requests.post(url, headers=headers, json=payload, timeout=self.llm_timeout)
        
        if response.status_code != 200:
            error_msg = (
                f"LLM request failed ({response.status_code}): {response.text}\n"
                f"Request URL: {url}\n"
                f"API Base: {self.llm_api_base}\n"
                f"Model: {self.llm_model}"
            )
            raise RuntimeError(error_msg)
        
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse JSON response
        try:
            scores = json.loads(message.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM response as JSON: {message}") from exc
        
        # Sleep if needed
        if self.llm_sleep > 0:
            time.sleep(self.llm_sleep)
        
        return scores
    
    def _call_llm_for_reranking(
        self,
        query: str,
        items: Sequence[Candidates],
    ) -> List[Dict[str, float]]:
        """
        Call LLM to re-score and re-rank items with automatic batching.
        
        Automatically batches items based on token limits to avoid exceeding
        the model's context window (default: 128k tokens for gpt-4o).
        
        Args:
            query: User query
            items: Items to re-score
            
        Returns:
            List of dicts with "item_id" and "score"
        """
        if not items:
            return []
        
        # Batch items based on token limits
        batches = self._batch_items_by_tokens(query, items)
        
        if len(batches) == 1:
            # Single batch, no need for batching
            LOGGER.debug("Single batch: %d items", len(items))
            return self._call_llm_for_single_batch(query, items)
        
        # Process multiple batches
        LOGGER.info("Processing %d items in %d batches", len(items), len(batches))
        all_scores: List[Dict[str, float]] = []
        
        for batch_idx, batch in enumerate(batches, 1):
            LOGGER.debug("Processing batch %d/%d: %d items", batch_idx, len(batches), len(batch))
            try:
                batch_scores = self._call_llm_for_single_batch(query, batch)
                all_scores.extend(batch_scores)
            except Exception as exc:
                LOGGER.error("Batch %d/%d failed: %s", batch_idx, len(batches), exc)
                raise
        
        return all_scores
    
    def search(
        self,
        query: str,
        query_id: str | None = None,
    ) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval with LLM re-ranking.
        
        Args:
            query: User query text
            query_id: Optional query ID for logging
            
        Returns:
            HybridRetrievalResult with top-5 and top-10 results
        """
        if not query.strip():
            return HybridRetrievalResult(
                query_id=query_id or "",
                query_text=query,
                top_5=[],
                top_10=[],
                bm25_results=[],
                vector_results=[],
                merged_items=[],
            )
        
        query_id = query_id or "unknown"
        
        # Step 1: BM25 retrieval (top-20)
        LOGGER.info("Query %s: BM25 retrieval (top-%d)", query_id, self.retrieval_top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=self.retrieval_top_k)
        LOGGER.debug("Query %s: BM25 retrieved %d results", query_id, len(bm25_results))
        
        # Step 2: Vector retrieval (top-20)
        LOGGER.info("Query %s: Vector retrieval (top-%d)", query_id, self.retrieval_top_k)
        vector_results = self.vector_retriever.search(query, top_k=self.retrieval_top_k)
        LOGGER.debug("Query %s: Vector retrieved %d results", query_id, len(vector_results))
        
        # Step 3: Merge and deduplicate
        LOGGER.info("Query %s: Merging and deduplicating results", query_id)
        merged_items = self._merge_and_deduplicate(bm25_results, vector_results)
        LOGGER.debug("Query %s: Merged to %d unique items", query_id, len(merged_items))
        
        if not merged_items:
            LOGGER.warning("Query %s: No items to re-rank", query_id)
            return HybridRetrievalResult(
                query_id=query_id,
                query_text=query,
                top_5=[],
                top_10=[],
                bm25_results=bm25_results,
                vector_results=vector_results,
                merged_items=[],
            )
        
        # Step 4: LLM re-scoring and re-ranking
        LOGGER.info("Query %s: LLM re-ranking %d items", query_id, len(merged_items))
        try:
            llm_scores = self._call_llm_for_reranking(query, merged_items)
        except Exception as exc:
            LOGGER.error("Query %s: LLM re-ranking failed: %s", query_id, exc)
            raise
        
        # Create a mapping from item_id to score
        score_map: Dict[str, float] = {}
        for score_dict in llm_scores:
            item_id = score_dict.get("item_id", "")
            if not item_id:
                continue
            try:
                score = float(score_dict.get("score", 0.0))
                score_map[item_id] = score
            except (ValueError, TypeError):
                LOGGER.warning("Invalid score for item_id %s: %s", item_id, score_dict.get("score"))
                continue
        
        # Check for missing scores
        missing_scores = [item.id for item in merged_items if item.id not in score_map]
        if missing_scores:
            LOGGER.warning(
                "Query %s: Missing LLM scores for %d items: %s",
                query_id, len(missing_scores), missing_scores[:5]
            )
        
        # Create list of (candidate, score) tuples and sort by score
        scored_items: List[Tuple[Candidates, float]] = []
        for item in merged_items:
            score = score_map.get(item.id, 0.0)  # Default to 0.0 if missing
            scored_items.append((item, score))
        
        # Sort by score (descending)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Return top-5 and top-10
        top_5 = scored_items[:5]
        top_10 = scored_items[:10]
        
        LOGGER.info(
            "Query %s: LLM re-ranking complete. Top-5 scores: %s",
            query_id,
            [f"{item.id}:{score:.1f}" for item, score in top_5]
        )
        
        return HybridRetrievalResult(
            query_id=query_id,
            query_text=query,
            top_5=top_5,
            top_10=top_10,
            bm25_results=bm25_results,
            vector_results=vector_results,
            merged_items=merged_items,
        )


def load_food_candidates_for_hybrid(csv_path: Path) -> List[Candidates]:
    """Load candidates for hybrid retrieval."""
    from src.bm25_retrieval import load_food_candidates
    return load_food_candidates(csv_path)

