"""Hybrid retrieval module combining BM25 and Vector retrieval with RRF fusion and LLM re-ranking.

This module implements a two-stage retrieval and re-ranking pipeline:
1. BM25 retrieval (top-K, configurable)
2. Vector retrieval (top-K, configurable)
3. RRF fusion (optional) or merge and deduplicate results
4. LLM re-scoring and re-ranking
5. Return top-K results (configurable, e.g., top-5 and top-10)

All parameters must be explicitly provided. No defaults are used.
Configuration is read from environment variables via .env file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import requests

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

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
    bm25_results: List[Tuple[Candidates, float]]  # Original BM25 results (top-50)
    vector_results: List[Tuple[Candidates, float]]  # Original vector results (top-50)
    merged_items: List[Candidates]  # Merged and deduplicated items before LLM scoring


class HybridRetriever:
    """
    Hybrid retriever that combines BM25 and Vector retrieval with LLM re-ranking.
    """
    
    def __init__(
        self,
        candidates: Sequence[Candidates],
        # LLM re-ranking parameters (required)
        llm_api_base: str,
        llm_api_key: str,
        llm_model: str,
        llm_max_tokens_per_item: int,
        llm_max_context_tokens: int,
        llm_reserved_output_tokens: int,
        llm_tokens_per_item_output: int,
        llm_sleep: float,
        llm_timeout: float,
        # BM25 parameters (required)
        bm25_k1: float,
        bm25_b: float,
        # Vector retrieval parameters (required)
        vector_api_base: str,
        vector_api_key: str,
        vector_model_name: str,
        vector_dimensions: int,
        vector_max_tokens_per_request: int,
        vector_max_items_per_batch: int | None,
        vector_rpm_limit: int,
        vector_timeout: float,
        vector_normalize_embeddings: bool,
        vector_hnsw_index_path: Path | str,  # Required: path to existing HNSW index file
        vector_use_hnsw: bool,
        vector_hnsw_m: int,
        vector_hnsw_ef_construction: int,
        vector_hnsw_ef_search: int,
        vector_embeddings_dir: Path | str | None,
        vector_cache_embeddings: bool,
        # Retrieval parameters (required)
        retrieval_top_k: int,  # Top-K for each retrieval method (default: 50)
        # RRF fusion parameters (required)
        use_rrf: bool,  # Whether to use RRF fusion before LLM re-ranking
        rrf_k: int,  # RRF constant k (typically 60)
        rrf_top_k: int,  # Top-K results after RRF fusion (input to LLM)
        # Final output parameters (required)
        final_top_k_1: int,  # First top-K result set (e.g., top-5)
        final_top_k_2: int,  # Second top-K result set (e.g., top-10)
    ) -> None:
        """
        Initialize hybrid retriever.
        
        All parameters must be explicitly provided. No defaults are used.
        
        Args:
            candidates: Sequence of candidate documents
            llm_api_base: LLM API base URL (required)
            llm_api_key: LLM API key (required)
            llm_model: LLM model name for re-ranking (required)
            llm_max_tokens_per_item: Maximum tokens allowed per item text (required)
            llm_max_context_tokens: Maximum context window tokens for the model (required)
            llm_reserved_output_tokens: Base reserved tokens for LLM response output (required)
            llm_tokens_per_item_output: Estimated tokens per item in output JSON (required)
            llm_sleep: Sleep time between LLM requests (required)
            llm_timeout: LLM request timeout (required)
            bm25_k1: BM25 k1 parameter (required)
            bm25_b: BM25 b parameter (required)
            vector_api_base: Vector API base URL (required)
            vector_api_key: Vector API key (required)
            vector_model_name: Vector model name (required)
            vector_dimensions: Vector embedding dimensions (required)
            vector_max_tokens_per_request: Max tokens per API request (required)
            vector_max_items_per_batch: Max items per batch (required, None for unlimited)
            vector_rpm_limit: Requests per minute limit (required)
            vector_timeout: Request timeout (required)
            vector_normalize_embeddings: Whether to normalize embeddings (required)
            vector_hnsw_index_path: Path to existing HNSW index file (required, must exist)
            vector_use_hnsw: Whether to use HNSW indexing (required)
            vector_hnsw_m: HNSW M parameter (required)
            vector_hnsw_ef_construction: HNSW ef_construction parameter (required)
            vector_hnsw_ef_search: HNSW ef_search parameter (required)
            vector_embeddings_dir: Directory for embeddings cache (required, None if not used)
            vector_cache_embeddings: Whether to cache embeddings (required)
            retrieval_top_k: Top-K results to retrieve from each method (required, typically 50)
            use_rrf: Whether to use RRF fusion before LLM re-ranking (required)
            rrf_k: RRF constant k (required, typically 60)
            rrf_top_k: Top-K results after RRF fusion (required, input to LLM, typically 50)
            final_top_k_1: First top-K result set after LLM re-ranking (required, e.g., 5)
            final_top_k_2: Second top-K result set after LLM re-ranking (required, e.g., 10)
        """
        self.candidates = list(candidates)
        self.retrieval_top_k = retrieval_top_k
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        self.rrf_top_k = rrf_top_k
        self.final_top_k_1 = final_top_k_1
        self.final_top_k_2 = final_top_k_2
        
        # Validate HNSW index path - must exist
        vector_hnsw_index_path_obj = Path(vector_hnsw_index_path)
        if not vector_hnsw_index_path_obj.exists():
            raise ValueError(
                f"HNSW index file does not exist: {vector_hnsw_index_path}\n"
                f"Please generate the index file first using vector_retrieval.py"
            )
        if not vector_hnsw_index_path_obj.is_file():
            raise ValueError(
                f"HNSW index path is not a file: {vector_hnsw_index_path}\n"
                f"Please provide the full path to the .index file"
            )
        
        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever(
            candidates,
            k1=bm25_k1,
            b=bm25_b,
        )
        
        # Initialize Vector retriever with all required parameters
        self.vector_retriever = VectorRetriever(
            candidates,
            api_base=vector_api_base,
            api_key=vector_api_key,
            model_name=vector_model_name,
            normalize_embeddings=vector_normalize_embeddings,
            max_tokens_per_request=vector_max_tokens_per_request,
            max_items_per_batch=vector_max_items_per_batch,
            rpm_limit=vector_rpm_limit,
            timeout=vector_timeout,
            dimensions=vector_dimensions,
            use_hnsw=vector_use_hnsw,
            index_path=vector_hnsw_index_path_obj,  # Use the validated path
            hnsw_m=vector_hnsw_m,
            hnsw_ef_construction=vector_hnsw_ef_construction,
            hnsw_ef_search=vector_hnsw_ef_search,
            embeddings_dir=Path(vector_embeddings_dir) if vector_embeddings_dir else None,
            cache_embeddings=vector_cache_embeddings,
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
    
    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[Candidates, float]],
        vector_results: List[Tuple[Candidates, float]],
    ) -> List[Candidates]:
        """
        Perform Reciprocal Rank Fusion (RRF) on BM25 and Vector results.
        
        RRF formula: RRF_score(d) = sum(1 / (k + rank_i(d)))
        where k is a constant (typically 60) and rank_i is the rank in list i.
        
        Args:
            bm25_results: BM25 retrieval results (sorted by score descending)
            vector_results: Vector retrieval results (sorted by score descending)
            
        Returns:
            List of top-K candidates after RRF fusion, sorted by RRF score descending
        """
        # Build rank maps: item_id -> rank (1-indexed)
        bm25_ranks: Dict[str, int] = {}
        for rank, (candidate, _) in enumerate(bm25_results, 1):
            bm25_ranks[candidate.id] = rank
        
        vector_ranks: Dict[str, int] = {}
        for rank, (candidate, _) in enumerate(vector_results, 1):
            vector_ranks[candidate.id] = rank
        
        # Collect all unique items
        all_item_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, Candidates] = {}
        
        for item_id in all_item_ids:
            # Get rank from each list (if present)
            bm25_rank = bm25_ranks.get(item_id)
            vector_rank = vector_ranks.get(item_id)
            
            # Calculate RRF score
            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + bm25_rank)
            if vector_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + vector_rank)
            
            rrf_scores[item_id] = rrf_score
            
            # Store candidate reference
            if item_id in bm25_ranks:
                # Find candidate from bm25_results
                for candidate, _ in bm25_results:
                    if candidate.id == item_id:
                        candidate_map[item_id] = candidate
                        break
            elif item_id in vector_ranks:
                # Find candidate from vector_results
                for candidate, _ in vector_results:
                    if candidate.id == item_id:
                        candidate_map[item_id] = candidate
                        break
        
        # Sort by RRF score (descending)
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-K candidates
        top_k_items = []
        for item_id, score in sorted_items[:self.rrf_top_k]:
            if item_id in candidate_map:
                top_k_items.append(candidate_map[item_id])
        
        LOGGER.debug(
            "RRF fusion: %d BM25 + %d Vector -> %d unique -> top-%d",
            len(bm25_results), len(vector_results), len(all_item_ids), len(top_k_items)
        )
        
        return top_k_items
    
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
            HybridRetrievalResult with configurable top-K results (top_5 and top_10 fields)
        """
        if not query.strip():
            return HybridRetrievalResult(
                query_id=query_id or "",
                query_text=query,
                top_5=[],  # Empty top_k_1
                top_10=[],  # Empty top_k_2
                bm25_results=[],
                vector_results=[],
                merged_items=[],
            )
        
        query_id = query_id or "unknown"
        
        # Step 1: BM25 retrieval (top-50)
        LOGGER.info("Query %s: BM25 retrieval (top-%d)", query_id, self.retrieval_top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=self.retrieval_top_k)
        LOGGER.debug("Query %s: BM25 retrieved %d results", query_id, len(bm25_results))
        
        # Step 2: Vector retrieval (top-50)
        LOGGER.info("Query %s: Vector retrieval (top-%d)", query_id, self.retrieval_top_k)
        vector_results = self.vector_retriever.search(query, top_k=self.retrieval_top_k)
        LOGGER.debug("Query %s: Vector retrieved %d results", query_id, len(vector_results))
        
        # Step 3: Merge and deduplicate (if not using RRF) or RRF fusion (if using RRF)
        if self.use_rrf:
            LOGGER.info("Query %s: RRF fusion (k=%d, top-%d)", query_id, self.rrf_k, self.rrf_top_k)
            rrf_items = self._rrf_fusion(bm25_results, vector_results)
            LOGGER.debug("Query %s: RRF fusion resulted in %d items", query_id, len(rrf_items))
            items_for_llm = rrf_items
        else:
            LOGGER.info("Query %s: Merging and deduplicating results", query_id)
            merged_items = self._merge_and_deduplicate(bm25_results, vector_results)
            LOGGER.debug("Query %s: Merged to %d unique items", query_id, len(merged_items))
            items_for_llm = merged_items
        
        if not items_for_llm:
            LOGGER.warning("Query %s: No items to re-rank", query_id)
            return HybridRetrievalResult(
                query_id=query_id,
                query_text=query,
                top_5=[],  # Empty top_k_1
                top_10=[],  # Empty top_k_2
                bm25_results=bm25_results,
                vector_results=vector_results,
                merged_items=[],  # Empty when no items to re-rank
            )
        
        # Step 4: LLM re-scoring and re-ranking
        LOGGER.info("Query %s: LLM re-ranking %d items", query_id, len(items_for_llm))
        try:
            llm_scores = self._call_llm_for_reranking(query, items_for_llm)
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
        missing_scores = [item.id for item in items_for_llm if item.id not in score_map]
        if missing_scores:
            LOGGER.warning(
                "Query %s: Missing LLM scores for %d items: %s",
                query_id, len(missing_scores), missing_scores[:5]
            )
        
        # Create list of (candidate, score) tuples and sort by score
        scored_items: List[Tuple[Candidates, float]] = []
        for item in items_for_llm:
            score = score_map.get(item.id, 0.0)  # Default to 0.0 if missing
            scored_items.append((item, score))
        
        # Sort by score (descending)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Return top-K results (configurable)
        top_k_1 = scored_items[:self.final_top_k_1]
        top_k_2 = scored_items[:self.final_top_k_2]
        
        LOGGER.info(
            "Query %s: LLM re-ranking complete. Top-%d scores: %s",
            query_id,
            self.final_top_k_1,
            [f"{item.id}:{score:.1f}" for item, score in top_k_1]
        )
        
        return HybridRetrievalResult(
            query_id=query_id,
            query_text=query,
            top_5=top_k_1,  # Stored as top_5 field for backward compatibility
            top_10=top_k_2,  # Stored as top_10 field for backward compatibility
            bm25_results=bm25_results,
            vector_results=vector_results,
            merged_items=items_for_llm,  # Store RRF fused items or merged items (always contains items sent to LLM)
        )


def load_food_candidates_for_hybrid(csv_path: Path) -> List[Candidates]:
    """Load candidates for hybrid retrieval."""
    from src.bm25_retrieval import load_food_candidates
    return load_food_candidates(csv_path)

