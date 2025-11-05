"""Lightweight reranker module for two-stage re-ranking.

This module provides a lightweight reranker (e.g., bge-reranker-base/small)
for the first stage of two-stage re-ranking, which reduces the candidate set
from 50 to 20 before LLM final scoring.

Optimizations:
- MPS device support (Mac Metal) with FP16 precision
- Dynamic padding for batch processing (handled by FlagReranker)
- Pre-tokenized input caching for document texts
- Concurrent batch processing
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

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
        batch_size: int = 32,
        # Pre-tokenization cache parameters
        tokenization_cache_dir: Path | str | None = None,
        tokenization_cache_enabled: bool = True,
        # Concurrent processing parameters
        max_concurrent_batches: int = 2,  # Number of batches to process concurrently
    ):
        """
        Initialize the lightweight reranker.
        
        Args:
            model_name: Model name (e.g., "BAAI/bge-reranker-base", "BAAI/bge-reranker-small")
            use_api: Whether to use API instead of local model
            api_base: API base URL (if using API)
            api_key: API key (if using API)
            device: Device to use ("cpu", "cuda", "mps", etc.). If None, auto-detect.
                On Mac, should be "mps" for Metal acceleration.
            batch_size: Batch size for processing (32 or 64 recommended for Mac MPS).
                FlagReranker handles dynamic padding automatically.
            tokenization_cache_dir: Directory to cache pre-tokenized item texts.
                If None, uses default cache directory.
            tokenization_cache_enabled: Whether to enable pre-tokenization caching.
            max_concurrent_batches: Number of batches to process concurrently (default: 2).
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_base = api_base
        self.api_key = api_key
        self.device = device
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        
        # Pre-tokenization cache setup
        self.tokenization_cache_enabled = tokenization_cache_enabled
        if tokenization_cache_dir:
            self.tokenization_cache_dir = Path(tokenization_cache_dir)
            self.tokenization_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default cache directory
            self.tokenization_cache_dir = Path("artifacts/reranker_tokenization_cache")
            self.tokenization_cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenized_items_cache: Dict[str, List[int]] = {}  # item_id -> token_ids
        self.tokenizer = None  # Will be initialized when model is loaded
        
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
                # Ensure MPS device is used on Mac (not CPU)
                if TORCH_AVAILABLE:
                    if device == "mps":
                        if not torch.backends.mps.is_available():
                            raise RuntimeError(
                                "MPS device requested but not available. "
                                "Requires macOS 12.3+ and PyTorch 1.12+. "
                                "Falling back to CPU is not recommended for performance."
                            )
                        torch.set_default_device("mps")
                        LOGGER.info("Using MPS device for reranker (Mac Metal acceleration)")
                        self.device = "mps"  # Store for later use
                    elif device == "cuda":
                        if not torch.cuda.is_available():
                            raise RuntimeError("CUDA device requested but not available")
                        torch.set_default_device("cuda")
                        LOGGER.info("Using CUDA device for reranker")
                        self.device = "cuda"
                    elif device is None:
                        # Auto-detect: prefer MPS on Mac, then CUDA, then CPU
                        if torch.backends.mps.is_available():
                            device = "mps"
                            torch.set_default_device("mps")
                            LOGGER.info("Auto-detected: Using MPS device for reranker (Mac Metal)")
                            self.device = "mps"
                        elif torch.cuda.is_available():
                            device = "cuda"
                            torch.set_default_device("cuda")
                            LOGGER.info("Auto-detected: Using CUDA device for reranker")
                            self.device = "cuda"
                        else:
                            device = "cpu"
                            LOGGER.warning("No GPU available, using CPU device for reranker (slow)")
                            self.device = "cpu"
                    else:
                        if device != "cpu":
                            LOGGER.warning(f"Device '{device}' may not be optimal. Using as specified.")
                        self.device = device
                else:
                    self.device = "cpu"
                
                # FlagReranker uses FP16 by default (use_fp16=True)
                # Ensure FP16 is explicitly enabled for better performance on MPS/CUDA
                # Note: torch_dtype=torch.float16 is handled internally by FlagReranker
                self.model = FlagReranker(model_name, use_fp16=True)
                
                # Initialize tokenizer for pre-tokenization caching
                if TRANSFORMERS_AVAILABLE and self.tokenization_cache_enabled:
                    try:
                        # Try to get tokenizer from FlagReranker's model if accessible
                        # Otherwise, initialize a new tokenizer using the same model name
                        if hasattr(self.model, 'tokenizer'):
                            self.tokenizer = self.model.tokenizer
                        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'tokenizer'):
                            self.tokenizer = self.model.model.tokenizer
                        else:
                            # Fallback: initialize tokenizer from model name
                            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        LOGGER.debug("Tokenizer initialized for pre-tokenization caching")
                    except Exception as e:
                        LOGGER.warning("Failed to initialize tokenizer for caching: %s. Caching disabled.", e)
                        self.tokenization_cache_enabled = False
                        self.tokenizer = None
                else:
                    self.tokenizer = None
                    if self.tokenization_cache_enabled and not TRANSFORMERS_AVAILABLE:
                        LOGGER.warning("transformers not available, pre-tokenization caching disabled")
                        self.tokenization_cache_enabled = False
                
                actual_device = self.device or (str(torch.get_default_device()) if TORCH_AVAILABLE else "unknown")
                LOGGER.info(
                    "Initialized Cross-Encoder reranker: %s (device: %s, batch_size: %d, fp16: True, max_concurrent_batches: %d, tokenization_cache: %s)",
                    model_name, actual_device, batch_size, max_concurrent_batches, 
                    "enabled" if self.tokenization_cache_enabled else "disabled"
                )
                
                # Load pre-tokenized cache if available
                if self.tokenization_cache_enabled:
                    self._load_tokenization_cache()
                
                # Initialize timing info storage
                self._last_timing_info = None
                
                # Warmup: Run a dummy reranking to initialize model internals and move to GPU if available
                # This significantly speeds up subsequent reranking operations
                LOGGER.debug("Warming up reranker model...")
                try:
                    warmup_start = time.time()
                    _ = self.model.compute_score([("warmup query", "warmup document")])
                    warmup_time = time.time() - warmup_start
                    LOGGER.debug("Reranker warmup completed in %.2fs", warmup_time)
                    if warmup_time > 2.0:
                        LOGGER.warning(
                            "Reranker warmup took %.2fs - model may be running on CPU. "
                            "Consider using GPU/MPS acceleration if available", warmup_time
                        )
                except Exception as e:
                    LOGGER.warning("Reranker warmup failed (non-critical): %s", e)
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
    
    def _get_cache_path(self) -> Path:
        """Get the path to the tokenization cache file."""
        # Create a hash of model name for cache file naming
        model_hash = hashlib.md5(self.model_name.encode()).hexdigest()[:8]
        return self.tokenization_cache_dir / f"tokenized_items_{model_hash}.pkl"
    
    def _load_tokenization_cache(self) -> None:
        """Load pre-tokenized items from cache."""
        if not self.tokenization_cache_enabled or not self.tokenization_cache_dir:
            return
        
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self.tokenized_items_cache = pickle.load(f)
                LOGGER.info("Loaded %d pre-tokenized items from cache: %s", 
                           len(self.tokenized_items_cache), cache_path)
            except Exception as e:
                LOGGER.warning("Failed to load tokenization cache: %s", e)
                self.tokenized_items_cache = {}
    
    def _save_tokenization_cache(self) -> None:
        """Save pre-tokenized items to cache."""
        if not self.tokenization_cache_enabled or not self.tokenization_cache_dir:
            return
        
        if not self.tokenized_items_cache:
            return
        
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.tokenized_items_cache, f)
            LOGGER.debug("Saved %d pre-tokenized items to cache: %s", 
                        len(self.tokenized_items_cache), cache_path)
        except Exception as e:
            LOGGER.warning("Failed to save tokenization cache: %s", e)
    
    def _pre_tokenize_items(self, items: Sequence[Tuple[str, str]]) -> Dict[str, List[int]]:
        """
        Pre-tokenize item texts and cache them.
        
        Args:
            items: List of (item_id, item_text) tuples
            
        Returns:
            Dict mapping item_id to token_ids
        """
        if not self.tokenization_cache_enabled or not self.tokenizer:
            return {}
        
        tokenized = {}
        for item_id, item_text in items:
            if item_id in self.tokenized_items_cache:
                # Already cached
                tokenized[item_id] = self.tokenized_items_cache[item_id]
            else:
                # Tokenize and cache
                try:
                    # Tokenize text (without query, just the document)
                    # FlagReranker typically uses format: query + [SEP] + document
                    # For caching, we only tokenize the document part
                    tokens = self.tokenizer.encode(
                        item_text,
                        add_special_tokens=False,  # Don't add [CLS] or [SEP] yet
                        return_attention_mask=False,
                        return_tensors=None,
                    )
                    tokenized[item_id] = tokens
                    self.tokenized_items_cache[item_id] = tokens
                except Exception as e:
                    LOGGER.warning("Failed to tokenize item %s: %s", item_id, e)
        
        # Save cache periodically (every 100 new items)
        if len(self.tokenized_items_cache) % 100 == 0:
            self._save_tokenization_cache()
        
        return tokenized
    
    def _build_pairs_with_cache(self, query: str, items: Sequence[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], float]:
        """
        Build query-document pairs, using cached tokenized documents if available.
        
        Note: FlagReranker's compute_score expects (query, document) text pairs,
        so we still need to provide text pairs. The caching is mainly useful for
        avoiding re-tokenization, but FlagReranker will handle the actual tokenization
        internally. This method prepares the pairs, and we can optimize by ensuring
        consistent text formatting.
        
        Args:
            query: User query
            items: List of (item_id, item_text) tuples
            
        Returns:
            Tuple of (pairs, time_taken) where pairs is List of (query, document_text) pairs
        """
        import time
        start_time = time.time()
        
        # For now, we'll build pairs normally. The pre-tokenization cache is mainly
        # useful if we want to implement custom batching with dynamic padding ourselves,
        # but FlagReranker already handles this efficiently.
        # We'll use the cache in a future optimization if needed.
        pairs = [(query, text) for _, text in items]
        
        # Pre-tokenize items for future use (even if not used now, cache them)
        if self.tokenization_cache_enabled:
            self._pre_tokenize_items(items)
        
        time_taken = time.time() - start_time
        return pairs, time_taken
    
    def _rerank_batches_concurrent(self, pairs: List[Tuple[str, str]]) -> Tuple[List[float], dict]:
        """
        Process batches concurrently for better performance.
        
        Args:
            pairs: List of (query, document) text pairs
            
        Returns:
            Tuple of (scores, timing_info) where:
            - scores: List of scores corresponding to each pair
            - timing_info: Dict with timing breakdown (build_pairs_time, batch_times, total_time, etc.)
        """
        import time
        timing_info = {
            'num_batches': 0,
            'batch_times': [],
            'total_inference_time': 0.0,
            'concurrent': False,
        }
        
        if len(pairs) <= self.batch_size:
            # Single batch, no need for concurrency
            batch_start = time.time()
            batch_scores = self.model.compute_score(pairs)
            batch_time = time.time() - batch_start
            timing_info['num_batches'] = 1
            timing_info['batch_times'] = [batch_time]
            timing_info['total_inference_time'] = batch_time
            timing_info['concurrent'] = False
            
            if isinstance(batch_scores, (int, float)):
                return [float(batch_scores)] * len(pairs), timing_info
            return [float(s) for s in batch_scores], timing_info
        
        # Split into batches
        batches = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batches.append(batch_pairs)
        
        timing_info['num_batches'] = len(batches)
        
        # Try concurrent processing first, but fall back to sequential if needed
        # MPS device may have issues with concurrent batch processing ("Already borrowed" error)
        # We'll try concurrent first, and if it fails, we'll catch the error and retry sequentially
        use_concurrent = self.max_concurrent_batches > 1
        timing_info['concurrent'] = use_concurrent
        
        LOGGER.debug("Processing %d batches (batch_size=%d, max_concurrent=%d, concurrent=%s)", 
                    len(batches), self.batch_size, self.max_concurrent_batches, use_concurrent)
        
        # Initialize all_scores
        all_scores = [0.0] * len(pairs)
        batch_times = [0.0] * len(batches)
        
        def process_batch(batch_idx: int, batch_pairs: List[Tuple[str, str]]) -> Tuple[int, List[float], float]:
            """Process a single batch and return scores with timing."""
            batch_start = time.time()
            batch_scores = self.model.compute_score(batch_pairs)
            batch_time = time.time() - batch_start
            
            if isinstance(batch_scores, (int, float)):
                batch_scores = [float(batch_scores)] * len(batch_pairs)
            else:
                batch_scores = [float(s) for s in batch_scores]
            return batch_idx, batch_scores, batch_time
        
        inference_start = time.time()
        
        if use_concurrent:
            # Try concurrent processing first
            # Note: MPS may have "Already borrowed" errors, so we'll catch and fall back to sequential
            try:
                with ThreadPoolExecutor(max_workers=self.max_concurrent_batches) as executor:
                    futures = {
                        executor.submit(process_batch, i, batch): i 
                        for i, batch in enumerate(batches)
                    }
                    
                    for future in as_completed(futures):
                        batch_idx = futures[future]
                        try:
                            idx, batch_scores, batch_time = future.result()
                            batch_times[idx] = batch_time
                            # Store scores in correct position
                            start_idx = idx * self.batch_size
                            for i, score in enumerate(batch_scores):
                                all_scores[start_idx + i] = score
                        except Exception as e:
                            # If any batch fails with "Already borrowed" or similar, fall back to sequential
                            error_msg = str(e).lower()
                            if "already borrowed" in error_msg or "mps" in error_msg:
                                LOGGER.warning("Concurrent processing failed (%s), falling back to sequential processing", e)
                                raise  # Re-raise to trigger fallback
                            else:
                                LOGGER.error("Batch %d processing failed: %s", batch_idx, e)
                                # Fill with dummy scores on error (non-recoverable)
                                start_idx = batch_idx * self.batch_size
                                end_idx = min(start_idx + self.batch_size, len(pairs))
                                for i in range(start_idx, end_idx):
                                    all_scores[i] = 0.0
            except Exception as e:
                # Fall back to sequential processing if concurrent processing fails
                error_msg = str(e).lower()
                if "already borrowed" in error_msg or (self.device == "mps" and "mps" in error_msg):
                    LOGGER.warning("Concurrent batch processing not supported (MPS limitation), using sequential processing")
                    timing_info['concurrent'] = False
                    # Reset and process sequentially
                    all_scores = [0.0] * len(pairs)
                    for batch_idx, batch in enumerate(batches):
                        try:
                            idx, batch_scores, batch_time = process_batch(batch_idx, batch)
                            batch_times[idx] = batch_time
                            start_idx = idx * self.batch_size
                            for i, score in enumerate(batch_scores):
                                all_scores[start_idx + i] = score
                        except Exception as batch_error:
                            LOGGER.error("Batch %d processing failed even sequentially: %s", batch_idx, batch_error)
                            start_idx = batch_idx * self.batch_size
                            end_idx = min(start_idx + self.batch_size, len(pairs))
                            for i in range(start_idx, end_idx):
                                all_scores[i] = 0.0
                else:
                    # Re-raise if it's not a recoverable error
                    raise
        else:
            # Sequential processing (when max_concurrent_batches=1)
            for batch_idx, batch in enumerate(batches):
                try:
                    idx, batch_scores, batch_time = process_batch(batch_idx, batch)
                    batch_times[idx] = batch_time
                    # Store scores in correct position
                    start_idx = idx * self.batch_size
                    for i, score in enumerate(batch_scores):
                        all_scores[start_idx + i] = score
                except Exception as e:
                    LOGGER.error("Batch %d processing failed: %s", batch_idx, e)
                    # Fill with dummy scores on error
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(pairs))
                    for i in range(start_idx, end_idx):
                        all_scores[i] = 0.0
        
        timing_info['total_inference_time'] = time.time() - inference_start
        timing_info['batch_times'] = batch_times
        
        return all_scores, timing_info
    
    def rerank(
        self,
        query: str,
        items: Sequence[Tuple[str, str]],  # List of (item_id, item_text) tuples
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Rerank items based on query relevance with batch processing.
        
        Args:
            query: User query
            items: List of (item_id, item_text) tuples to rerank
            top_k: Number of top items to return
            
        Returns:
            List of (item_id, score) tuples, sorted by score (descending)
        """
        if not items:
            return []
        
        # Only use dummy scores if we have significantly fewer items than top_k
        # This avoids returning dummy scores when we want actual scores for all items
        if len(items) < top_k and top_k > len(items) * 2:
            # If we have much fewer items than top_k (e.g., 5 items but top_k=20),
            # return all with dummy scores to avoid unnecessary computation
            LOGGER.debug("Only %d items (much fewer than top_k=%d), returning all without reranking", len(items), top_k)
            return [(item_id, 1.0) for item_id, _ in items]
        
        # Build pairs (with caching support) - track timing
        pairs, build_pairs_time = self._build_pairs_with_cache(query, items)
        
        if self.use_api:
            scores = self._rerank_via_api(pairs)
            timing_info = {'build_pairs_time': build_pairs_time, 'total_inference_time': 0.0}
        else:
            # Ensure model is initialized (should already be done in __init__)
            # This check ensures we don't include initialization time in reranking time
            if self.model is None:
                raise RuntimeError(
                    "Reranker model not initialized. This should not happen - "
                    "model should be initialized in __init__, not during reranking."
                )
            
            # Process in batches with concurrent execution
            # Note: Model initialization time is NOT included here - it's done in __init__
            # FlagReranker handles dynamic padding automatically based on batch contents
            scores, batch_timing_info = self._rerank_batches_concurrent(pairs)
            timing_info = {
                'build_pairs_time': build_pairs_time,
                **batch_timing_info,
            }
        
        # Store timing info for later retrieval
        self._last_timing_info = timing_info
        
        # Handle both single score and list of scores (fallback for non-batched)
        if isinstance(scores, (int, float)):
            # Single score (shouldn't happen with multiple pairs, but handle it)
            scores = [float(scores)] * len(pairs)
        else:
            scores = [float(s) for s in scores]
        
        # Verify batch_size is being used correctly
        if len(items) > self.batch_size:
            num_batches = (len(items) + self.batch_size - 1) // self.batch_size
            LOGGER.debug("Processed %d items in %d batches (batch_size=%d)", 
                        len(items), num_batches, self.batch_size)
        
        # Create list of (item_id, score) tuples
        scored_items = [(item_id, score) for (item_id, _), score in zip(items, scores)]
        
        # Sort by score (descending) and return top_k
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        if scored_items:
            score_range = (
                scored_items[0][1],
                scored_items[min(top_k-1, len(scored_items)-1)][1] if len(scored_items) >= top_k else scored_items[-1][1]
            )
        else:
            score_range = (0.0, 0.0)
        
        LOGGER.info(
            "Reranked %d items to top-%d: score range [%.4f, %.4f] (batch_size: %d, device: %s)",
            len(items), top_k, score_range[0], score_range[1], self.batch_size, self.device or "auto"
        )
        
        return scored_items[:top_k]
    
    def get_last_timing_info(self) -> dict | None:
        """
        Get the timing breakdown from the last rerank call.
        
        Returns:
            Dict with timing information, or None if no rerank has been called yet.
            Keys include:
            - build_pairs_time: Time to build query-document pairs
            - total_inference_time: Total time for model inference
            - num_batches: Number of batches processed
            - batch_times: List of individual batch processing times
            - concurrent: Whether concurrent processing was used
        """
        return getattr(self, '_last_timing_info', None)
    
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
    
    def save_cache(self) -> None:
        """Manually save tokenization cache (useful for cleanup)."""
        self._save_tokenization_cache()
