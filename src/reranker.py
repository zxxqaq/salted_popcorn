"""Lightweight reranker module for two-stage re-ranking.

This module provides a lightweight reranker (e.g., bge-reranker-base/small)
for the first stage of two-stage re-ranking, which reduces the candidate set
from 50 to 20 before LLM final scoring.
"""

from __future__ import annotations

import logging
import os
from typing import List, Sequence, Tuple

LOGGER = logging.getLogger("reranker")

try:
    from FlagEmbedding import FlagReranker
    FLAG_RERANKER_AVAILABLE = True
except ImportError:
    FLAG_RERANKER_AVAILABLE = False
    LOGGER.warning(
        "FlagReranker not available. Install with: pip install FlagEmbedding"
    )

try:
    import requests
except ImportError:
    requests = None


class LightweightReranker:
    """
    Lightweight reranker for first-stage re-ranking.
    
    Uses bge-reranker-base or bge-reranker-small to quickly filter
    candidates from 50 to 20 before LLM final scoring.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        use_api: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
    ):
        """
        Initialize the lightweight reranker.
        
        Args:
            model_name: Model name (e.g., "BAAI/bge-reranker-base", "BAAI/bge-reranker-small")
            use_api: Whether to use API instead of local model
            api_base: API base URL (if using API)
            api_key: API key (if using API)
            device: Device to use ("cpu", "cuda", etc.). If None, auto-detect.
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_base = api_base
        self.api_key = api_key
        self.device = device
        
        if use_api:
            if not api_base:
                raise ValueError("api_base is required when use_api=True")
            self.model = None
            LOGGER.info("Using API-based reranker: %s", api_base)
        else:
            if not FLAG_RERANKER_AVAILABLE:
                raise ImportError(
                    "FlagReranker is not available. Install with: pip install FlagEmbedding"
                )
            try:
                self.model = FlagReranker(model_name, use_fp16=True)
                if device:
                    # FlagReranker doesn't have explicit device parameter,
                    # but we can log it
                    LOGGER.info("Initialized local reranker: %s (device: %s)", model_name, device)
                else:
                    LOGGER.info("Initialized local reranker: %s", model_name)
            except Exception as e:
                error_msg = str(e)
                # Provide helpful suggestions for common errors
                suggestions = []
                if "not a valid model identifier" in error_msg or "Repository Not Found" in error_msg:
                    suggestions.append(
                        f"Model '{model_name}' not found. Common BGE reranker models:\n"
                        f"  - BAAI/bge-reranker-base (default, recommended)\n"
                        f"  - BAAI/bge-reranker-large\n"
                        f"  - BAAI/bge-reranker-v2-m3"
                    )
                if "401" in error_msg or "permission" in error_msg.lower() or "token" in error_msg.lower():
                    suggestions.append(
                        "If this is a private/gated model, you need to authenticate:\n"
                        "  1. Run: huggingface-cli login\n"
                        "  2. Or set HF_TOKEN environment variable"
                    )
                
                full_error = f"Failed to initialize reranker model {model_name}: {error_msg}"
                if suggestions:
                    full_error += "\n\n" + "\n\n".join(suggestions)
                
                raise RuntimeError(full_error) from e
    
    def rerank(
        self,
        query: str,
        items: Sequence[Tuple[str, str]],  # List of (item_id, item_text) tuples
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Rerank items based on query relevance.
        
        Args:
            query: User query
            items: List of (item_id, item_text) tuples to rerank
            top_k: Number of top items to return
            
        Returns:
            List of (item_id, score) tuples, sorted by score (descending)
        """
        if not items:
            return []
        
        if len(items) <= top_k:
            # If we have fewer items than top_k, return all with dummy scores
            LOGGER.debug("Only %d items, returning all without reranking", len(items))
            return [(item_id, 1.0) for item_id, _ in items]
        
        pairs = [(query, text) for _, text in items]
        
        if self.use_api:
            scores = self._rerank_via_api(pairs)
        else:
            scores = self.model.compute_score(pairs)
        
        # Handle both single score and list of scores
        if isinstance(scores, (int, float)):
            # Single score (shouldn't happen with multiple pairs, but handle it)
            scores = [float(scores)] * len(pairs)
        else:
            scores = [float(s) for s in scores]
        
        # Create list of (item_id, score) tuples
        scored_items = [(item_id, score) for (item_id, _), score in zip(items, scores)]
        
        # Sort by score (descending) and return top_k
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        LOGGER.info(
            "Reranked %d items to top-%d: score range [%.4f, %.4f]",
            len(items), top_k, scored_items[0][1] if scored_items else 0,
            scored_items[top_k-1][1] if len(scored_items) >= top_k else (scored_items[-1][1] if scored_items else 0)
        )
        
        return scored_items[:top_k]
    
    def _rerank_via_api(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Rerank via API (if API support is implemented)."""
        # TODO: Implement API-based reranking if needed
        # For now, raise an error
        raise NotImplementedError("API-based reranking not yet implemented")
    
    def rerank_candidates(
        self,
        query: str,
        candidates: Sequence,  # List of Candidates objects
        top_k: int = 20,
    ) -> List:
        """
        Rerank Candidates objects.
        
        Args:
            query: User query
            candidates: List of Candidates objects
            top_k: Number of top items to return
            
        Returns:
            List of Candidates objects (top_k items)
        """
        # Convert Candidates to (item_id, text) tuples
        items = [(c.id, c.text) for c in candidates]
        
        # Rerank
        scored_items = self.rerank(query, items, top_k=top_k)
        
        # Create mapping from item_id to Candidates
        candidate_map = {c.id: c for c in candidates}
        
        # Return top_k Candidates objects
        top_candidates = [candidate_map[item_id] for item_id, _ in scored_items]
        
        return top_candidates

