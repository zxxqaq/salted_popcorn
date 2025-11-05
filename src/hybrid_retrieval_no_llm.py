"""Hybrid retrieval module combining BM25 and Vector retrieval with RRF fusion and Cross-Encoder re-ranking.

This module implements a retrieval and re-ranking pipeline:
1. BM25 retrieval (top-K, configurable)
2. Vector retrieval (top-K, configurable)
3. RRF fusion (optional) or merge and deduplicate results
4. Cross-Encoder re-ranking (using BAAI/bge-reranker-base)
5. Return top-K results (configurable, e.g., top-5 and top-10)

All parameters must be explicitly provided. No defaults are used.
Configuration is read from environment variables via .env file.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

from src.bm25_retrieval import BM25Retriever, Candidates
from src.vector_retrieval import VectorRetriever

# Cross-Encoder reranker for final re-ranking
try:
    from src.reranker import LightweightReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    LightweightReranker = None

LOGGER = logging.getLogger("hybrid_retrieval_no_llm")


@dataclass
class HybridRetrievalResult:
    """Result from hybrid retrieval."""
    query_id: str
    query_text: str
    top_5: List[Tuple[Candidates, float]]  # (candidate, reranker_score)
    top_10: List[Tuple[Candidates, float]]  # (candidate, reranker_score)
    bm25_results: List[Tuple[Candidates, float]]  # Original BM25 results (top-50)
    vector_results: List[Tuple[Candidates, float]]  # Original vector results (top-50)
    merged_items: List[Candidates]  # Merged items before Cross-Encoder re-ranking
    # Timing information (in seconds)
    bm25_time: float = 0.0
    vector_time: float = 0.0
    rrf_time: float = 0.0  # or merge_time if not using RRF
    rerank_time: float = 0.0
    total_time: float = 0.0


class HybridRetriever:
    """
    Hybrid retriever that combines BM25 and Vector retrieval with Cross-Encoder re-ranking.
    """
    
    def __init__(
        self,
        candidates: Sequence[Candidates],
        # BM25 parameters (required)
        bm25_k1: float,
        bm25_b: float,
        # Vector retrieval parameters (choose one: API or local model)
        # Required parameters (no defaults)
        vector_hnsw_index_path: Path | str,  # Required: path to existing HNSW index file
        # Retrieval parameters (required)
        retrieval_top_k: int,  # Top-K for each retrieval method (default: 50)
        # RRF fusion parameters (required)
        use_rrf: bool,  # Whether to use RRF fusion before Cross-Encoder re-ranking
        rrf_k: int,  # RRF constant k (typically 60)
        rrf_top_k: int,  # Top-K results after RRF fusion (input to Cross-Encoder)
        # Final output parameters (required)
        final_top_k_1: int,  # First top-K result set (e.g., top-5)
        final_top_k_2: int,  # Second top-K result set (e.g., top-10)
        # Cross-Encoder re-ranking parameters (required)
        reranker_model: str,  # Model name for Cross-Encoder reranker (e.g., "BAAI/bge-reranker-base")
        # Optional parameters (with defaults)
        # API parameters (optional, if using API)
        vector_api_base: str | None = None,
        vector_api_key: str | None = None,
        vector_model_name: str | None = None,  # API model name (e.g., "text-embedding-3-small")
        vector_dimensions: int | None = None,  # API model dimensions
        vector_max_tokens_per_request: int = 8192,
        vector_max_items_per_batch: int | None = None,
        vector_rpm_limit: int = 300,
        vector_timeout: float = 120.0,
        # Local model parameters (optional, if using local model)
        vector_local_model_name: str | None = None,  # Local SentenceTransformer model (e.g., "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vector_normalize_embeddings: bool = True,
        vector_use_hnsw: bool = True,
        vector_hnsw_m: int = 32,
        vector_hnsw_ef_construction: int = 100,
        vector_hnsw_ef_search: int = 64,
        vector_embeddings_dir: Path | str | None = None,
        vector_cache_embeddings: bool = True,
        # Cross-Encoder re-ranking optional parameters
        reranker_device: str | None = None,  # Device to use ("mps", "cuda", "cpu", or None for auto)
        reranker_batch_size: int = 32,  # Batch size for Cross-Encoder (32 or 64 recommended for Mac MPS)
        reranker_top_k: int | None = None,  # Top-K items to return from reranker (None = return all scored items)
        reranker_tokenization_cache_dir: Path | str | None = None,  # Directory for pre-tokenization cache
        reranker_tokenization_cache_enabled: bool = True,  # Enable pre-tokenization caching
        reranker_max_concurrent_batches: int = 2,  # Number of batches to process concurrently
        # Query embedding parameters (optional, must come after all required parameters)
        vector_query_embedding_model: str | None = None,  # Optional: Local SentenceTransformer model for query embeddings
        # BM25 caching parameters (optional, must come after all required parameters)
        bm25_cache_dir: Path | str | None = None,
        bm25_cache_enabled: bool = True,
        bm25_data_source_hash: str | None = None,
    ) -> None:
        """
        Initialize hybrid retriever.
        
        All parameters must be explicitly provided. No defaults are used.
        
        Args:
            candidates: Sequence of candidate documents
            bm25_k1: BM25 k1 parameter (required)
            bm25_b: BM25 b parameter (required)
            vector_api_base: Vector API base URL (optional, if using API)
            vector_api_key: Vector API key (optional, if using API)
            vector_model_name: Vector API model name (optional, if using API, e.g., "text-embedding-3-small")
            vector_dimensions: Vector API model dimensions (optional, if using API)
            vector_max_tokens_per_request: Max tokens per API request (optional, only for API mode)
            vector_max_items_per_batch: Max items per batch (optional, only for API mode, None for unlimited)
            vector_rpm_limit: Requests per minute limit (optional, only for API mode)
            vector_timeout: Request timeout (optional, only for API mode)
            vector_local_model_name: Local SentenceTransformer model name (optional, if using local model, e.g., "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            vector_normalize_embeddings: Whether to normalize embeddings (required)
            vector_hnsw_index_path: Path to existing HNSW index file (required, must exist)
            vector_use_hnsw: Whether to use HNSW indexing (required)
            vector_hnsw_m: HNSW M parameter (required)
            vector_hnsw_ef_construction: HNSW ef_construction parameter (required)
            vector_hnsw_ef_search: HNSW ef_search parameter (required)
            vector_embeddings_dir: Directory for embeddings cache (optional, None if not used)
            vector_cache_embeddings: Whether to cache embeddings (required)
            vector_query_embedding_model: Optional local SentenceTransformer model for query embeddings (defaults to vector_local_model_name if not set)
            retrieval_top_k: Top-K results to retrieve from each method (required, typically 50)
            use_rrf: Whether to use RRF fusion before Cross-Encoder re-ranking (required)
            rrf_k: RRF constant k (required, typically 60)
            rrf_top_k: Top-K results after RRF fusion (required, input to Cross-Encoder, typically 50)
            final_top_k_1: First top-K result set after Cross-Encoder re-ranking (required, e.g., 5)
            final_top_k_2: Second top-K result set after Cross-Encoder re-ranking (required, e.g., 10)
            reranker_model: Model name for Cross-Encoder reranker (required)
            reranker_device: Device to use for Cross-Encoder ("mps", "cuda", "cpu", or None for auto)
            reranker_batch_size: Batch size for Cross-Encoder processing (required, 32 or 64 recommended)
            reranker_top_k: Top-K items to return from reranker (None = return all scored items, recommended: set to final_top_k_2 or larger)
        """
        self.candidates = list(candidates)
        self.retrieval_top_k = retrieval_top_k
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        self.rrf_top_k = rrf_top_k
        self.final_top_k_1 = final_top_k_1
        self.final_top_k_2 = final_top_k_2
        self.reranker_top_k = reranker_top_k
        
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
            cache_dir=bm25_cache_dir,
            cache_enabled=bm25_cache_enabled,
            data_source_hash=bm25_data_source_hash,
        )
        
        # Initialize Vector retriever with all required parameters
        # Determine if using API or local model
        if vector_local_model_name:
            # Use local model
            self.vector_retriever = VectorRetriever(
                candidates,
                local_model_name=vector_local_model_name,
                normalize_embeddings=vector_normalize_embeddings,
                use_hnsw=vector_use_hnsw,
                index_path=vector_hnsw_index_path_obj,  # Use the validated path
                hnsw_m=vector_hnsw_m,
                hnsw_ef_construction=vector_hnsw_ef_construction,
                hnsw_ef_search=vector_hnsw_ef_search,
                embeddings_dir=Path(vector_embeddings_dir) if vector_embeddings_dir else None,
                cache_embeddings=vector_cache_embeddings,
                query_embedding_model=vector_query_embedding_model,  # Optional: local model for query embeddings (defaults to local_model_name)
            )
        elif vector_api_key:
            # Use API
            self.vector_retriever = VectorRetriever(
                candidates,
                api_base=vector_api_base or "https://api.openai.com/v1",
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
                query_embedding_model=vector_query_embedding_model,  # Optional: local model for query embeddings
            )
        else:
            raise ValueError(
                "Either vector_local_model_name or vector_api_key must be provided for vector retrieval."
            )
        
        # Initialize Cross-Encoder reranker
        if not RERANKER_AVAILABLE:
            raise ImportError(
                "Cross-Encoder reranker is required. Install with: pip install FlagEmbedding"
            )
        try:
            # Initialize Cross-Encoder reranker
            # Note: LightweightReranker.__init__ will log the initialization details
            self.reranker = LightweightReranker(
                model_name=reranker_model,
                use_api=False,  # Use local model
                device=reranker_device,  # "mps", "cuda", "cpu", or None
                batch_size=reranker_batch_size,  # 32 or 64 for Mac MPS
                tokenization_cache_dir=reranker_tokenization_cache_dir,
                tokenization_cache_enabled=reranker_tokenization_cache_enabled,
                max_concurrent_batches=reranker_max_concurrent_batches,
            )
            # No need to log here - LightweightReranker.__init__ already logs initialization
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Cross-Encoder reranker: {e}"
            ) from e
    
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
    
    def search(
        self,
        query: str,
        query_id: str | None = None,
    ) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval with Cross-Encoder re-ranking.
        
        Args:
            query: User query text
            query_id: Optional query ID for logging
            
        Returns:
            HybridRetrievalResult with configurable top-K results (top_5 and top_10 fields)
        """
        query_id = query_id or "unknown"
        total_start = time.time()
        
        if not query.strip():
            return HybridRetrievalResult(
                query_id=query_id,
                query_text=query,
                top_5=[],  # Empty top_k_1
                top_10=[],  # Empty top_k_2
                bm25_results=[],
                vector_results=[],
                merged_items=[],
                total_time=time.time() - total_start,
            )
        
        # Step 1 & 2: Concurrent BM25 and Vector retrieval (top-50)
        LOGGER.info("Query %s: Starting concurrent retrieval (BM25 + Vector, top-%d)", query_id, self.retrieval_top_k)
        retrieval_start = time.time()
        
        def run_bm25_search():
            """Run BM25 search and return results with timing."""
            start = time.time()
            results = self.bm25_retriever.search(query, top_k=self.retrieval_top_k)
            elapsed = time.time() - start
            return results, elapsed
        
        def run_vector_search():
            """Run Vector search and return results with timing."""
            start = time.time()
            results = self.vector_retriever.search(query, top_k=self.retrieval_top_k)
            elapsed = time.time() - start
            return results, elapsed
        
        # Execute both retrievals concurrently
        bm25_results = []
        vector_results = []
        bm25_time = 0.0
        vector_time = 0.0
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            bm25_future = executor.submit(run_bm25_search)
            vector_future = executor.submit(run_vector_search)
            
            # Wait for both to complete and collect results
            futures = {bm25_future: "bm25", vector_future: "vector"}
            for future in as_completed(futures):
                task_type = futures[future]
                try:
                    results, elapsed = future.result()
                    if task_type == "bm25":
                        bm25_results = results
                        bm25_time = elapsed
                        LOGGER.info("Query %s: BM25 retrieved %d results in %.2fs", query_id, len(bm25_results), bm25_time)
                    else:  # vector
                        vector_results = results
                        vector_time = elapsed
                        LOGGER.info("Query %s: Vector retrieved %d results in %.2fs", query_id, len(vector_results), vector_time)
                        # Note: Vector retrieval's print statements (query embedding, HNSW search) 
                        # may appear after this log due to output buffering in concurrent threads.
                        # This is normal and doesn't affect functionality.
                except Exception as e:
                    LOGGER.error("Query %s: %s retrieval failed: %s", query_id, task_type.upper(), e)
                    if task_type == "bm25":
                        bm25_results = []
                        bm25_time = 0.0
                    else:
                        vector_results = []
                        vector_time = 0.0
        
        retrieval_elapsed = time.time() - retrieval_start
        LOGGER.info("Query %s: Concurrent retrieval completed in %.2fs (BM25: %.2fs, Vector: %.2fs, saved: %.2fs)", 
                    query_id, retrieval_elapsed, bm25_time, vector_time, 
                    max(bm25_time, vector_time) - retrieval_elapsed)
        
        if len(bm25_results) < self.retrieval_top_k:
            LOGGER.warning(
                "Query %s: BM25 only returned %d results (requested %d). This may indicate:",
                query_id, len(bm25_results), self.retrieval_top_k
            )
            LOGGER.warning(
                "  - Low matching score threshold (all results below threshold)"
            )
            LOGGER.warning(
                "  - Query terms not matching document vocabulary"
            )
            LOGGER.warning(
                "  - Tokenization issues (especially for non-English queries)"
            )
        
        # Step 3: Merge and deduplicate (if not using RRF) or RRF fusion (if using RRF)
        rrf_time = 0.0
        merge_time = 0.0
        if self.use_rrf:
            LOGGER.info("Query %s: RRF fusion (k=%d, top-%d)", query_id, self.rrf_k, self.rrf_top_k)
            rrf_start = time.time()
            items_for_reranking = self._rrf_fusion(bm25_results, vector_results)
            rrf_time = time.time() - rrf_start
            LOGGER.info(
                "Query %s: RRF fusion completed in %.2fs: %d items",
                query_id, rrf_time, len(items_for_reranking)
            )
        else:
            LOGGER.info("Query %s: Merging and deduplicating results", query_id)
            merge_start = time.time()
            merged_items = self._merge_and_deduplicate(bm25_results, vector_results)
            merge_time = time.time() - merge_start
            LOGGER.info(
                "Query %s: Merged to %d unique items in %.2fs",
                query_id, len(merged_items), merge_time
            )
            items_for_reranking = merged_items
        
        if not items_for_reranking:
            LOGGER.warning("Query %s: No items to re-rank", query_id)
            total_time = time.time() - total_start
            return HybridRetrievalResult(
                query_id=query_id,
                query_text=query,
                top_5=[],  # Empty top_k_1
                top_10=[],  # Empty top_k_2
                bm25_results=bm25_results,
                vector_results=vector_results,
                merged_items=[],  # Empty when no items to re-rank
                bm25_time=bm25_time,
                vector_time=vector_time,
                rrf_time=rrf_time if self.use_rrf else merge_time,
                total_time=total_time,
            )
        
        # Step 4: Cross-Encoder re-ranking (50 → top-10)
        LOGGER.info(
            "Query %s: Cross-Encoder re-ranking %d items (target: top-%d)",
            query_id, len(items_for_reranking), self.final_top_k_2
        )
        rerank_start = time.time()
        try:
            # Get scores for items in one call (avoids dummy score issue)
            # Use reranker_top_k if set, otherwise score all items
            # If reranker_top_k is set, only return top-K items (more efficient)
            # If reranker_top_k is None, score all items (needed if final_top_k_1 != final_top_k_2)
            rerank_top_k = self.reranker_top_k if self.reranker_top_k is not None else len(items_for_reranking)
            # Ensure rerank_top_k is at least as large as final_top_k_2 to get all needed results
            rerank_top_k = max(rerank_top_k, self.final_top_k_2)
            
            items_with_scores_all = self.reranker.rerank(
                query=query,
                items=[(item.id, item.text) for item in items_for_reranking],
                top_k=rerank_top_k,  # Get top-K scores (or all if reranker_top_k is None)
            )
            rerank_time = time.time() - rerank_start
            LOGGER.info(
                "Query %s: Cross-Encoder re-ranking completed in %.2fs (%d items scored)",
                query_id, rerank_time, len(items_for_reranking)
            )
        except Exception as exc:
            rerank_time = time.time() - rerank_start
            LOGGER.error(
                "Query %s: Cross-Encoder re-ranking failed after %.2fs: %s",
                query_id, rerank_time, exc
            )
            raise
        
        # Create mapping from item_id to candidate
        candidate_map = {item.id: item for item in items_for_reranking}
        
        # Create list of (candidate, score) tuples from all scored items
        # items_with_scores_all is already sorted by score (descending)
        scored_items: List[Tuple[Candidates, float]] = [
            (candidate_map[item_id], score)
            for item_id, score in items_with_scores_all
            if item_id in candidate_map
        ]
        
        # Ensure sorted by score (descending) - should already be sorted from rerank()
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Return top-K results (configurable)
        top_k_1 = scored_items[:self.final_top_k_1]
        top_k_2 = scored_items[:self.final_top_k_2]
        
        if top_k_1:
            score_range_1 = (top_k_1[0][1], top_k_1[-1][1])
        else:
            score_range_1 = (0.0, 0.0)
        
        if top_k_2:
            score_range_2 = (top_k_2[0][1], top_k_2[-1][1])
        else:
            score_range_2 = (0.0, 0.0)
        
        LOGGER.info(
            "Query %s: Cross-Encoder re-ranking complete. Top-%d scores: [%.4f, %.4f], Top-%d scores: [%.4f, %.4f]",
            query_id,
            self.final_top_k_1, score_range_1[0], score_range_1[1],
            self.final_top_k_2, score_range_2[0], score_range_2[1],
        )
        
        total_time = time.time() - total_start
        
        return HybridRetrievalResult(
            query_id=query_id,
            query_text=query,
            top_5=top_k_1,  # Stored as top_5 field for backward compatibility
            top_10=top_k_2,  # Stored as top_10 field for backward compatibility
            bm25_results=bm25_results,
            vector_results=vector_results,
            merged_items=items_for_reranking,  # Store items before Cross-Encoder re-ranking
            bm25_time=bm25_time,
            vector_time=vector_time,
            rrf_time=rrf_time if self.use_rrf else merge_time,
            rerank_time=rerank_time,
            total_time=total_time,
        )


def load_food_candidates_for_hybrid(
    csv_path: Path,
    cache_dir: Path | str | None = None,
    cache_enabled: bool = True,
) -> tuple[List[Candidates], str]:
    """Load candidates for hybrid retrieval with caching support."""
    from src.bm25_retrieval import load_food_candidates
    return load_food_candidates(csv_path, cache_dir=cache_dir, cache_enabled=cache_enabled)
