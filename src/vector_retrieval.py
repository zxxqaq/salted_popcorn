from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

try:
    import requests
except ImportError:
    requests = None  # Optional if using openai library

try:
    import openai
except ImportError:
    openai = None  # Optional if using requests directly

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required for token calculation. "
        "Install it with: pip install tiktoken"
    )


try:
    import hnswlib
except ImportError:
    hnswlib = None  # Optional for HNSW indexing

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # Optional for local query embedding
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import shared utilities
from src.bm25_retrieval import Candidates, load_food_candidates


# ---------------------------------------------------------------------------
# Token calculation utilities
# ---------------------------------------------------------------------------


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    
    For text-embedding-3-small and text-embedding-3-large, use "cl100k_base" encoding.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def estimate_batch_tokens(texts: Sequence[str], encoding_name: str = "cl100k_base") -> int:
    """Estimate total token count for a batch of texts."""
    encoding = tiktoken.get_encoding(encoding_name)
    total_tokens = 0
    for text in texts:
        total_tokens += len(encoding.encode(text))
    return total_tokens


def truncate_text_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        encoding_name: Tiktoken encoding name
        
    Returns:
        Truncated text that fits within token limit
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    
    return truncated_text


def batch_by_tokens(
    texts: Sequence[str],
    max_tokens_per_request: int = 8000000,  # Max tokens per API request (~8M, conservative limit)
    max_items_per_batch: int | None = None,  # Optional: Max items per request (only for very short texts)
    encoding_name: str = "cl100k_base",
    model_name: str | None = None
) -> Iterator[List[str]]:
    """
    Split texts into batches based on token count limits for API requests.
    
    Each batch will be sent as a single API request. The function ensures that
    the total tokens in each batch does not exceed the limit.
    
    Args:
        texts: List of text strings to batch
        max_tokens_per_request: Maximum total tokens per API request (default 8M, safe limit)
        max_items_per_batch: Optional maximum number of items per request.
            Only relevant when texts are very short (many items fit within token limit).
            If None, only token limit is used. Default None (no limit).
        encoding_name: Tiktoken encoding name (default: "cl100k_base" for embedding models)
        model_name: Optional model name to auto-detect encoding
        
    Yields:
        Batches of texts, each batch = one API request, respecting token and item count limits
        
    Example:
        >>> texts = ["text1", "text2", "text3", ...]
        >>> for batch in batch_by_tokens(texts, max_tokens_per_request=8000):
        ...     # Each batch is sent as one API request
        ...     embeddings = call_api(batch)
    """
    # Auto-detect encoding based on model if provided
    if model_name:
        if "text-embedding-3" in model_name:
            encoding_name = "cl100k_base"
        elif "text-embedding-ada-002" in model_name:
            encoding_name = "cl100k_base"
        # Add more model mappings as needed
    
    encoding = tiktoken.get_encoding(encoding_name)
    current_batch: List[str] = []
    current_batch_tokens = 0
    
    for text in texts:
        # Calculate tokens for this text
        text_tokens = len(encoding.encode(text))
        
        # If single text exceeds max_tokens, truncate it
        if text_tokens > max_tokens_per_request:
            import warnings
            warnings.warn(
                f"Text exceeds max_tokens_per_request ({max_tokens_per_request}): "
                f"text has {text_tokens} tokens. Truncating to fit."
            )
            # Truncate text to fit within limit
            text = truncate_text_to_tokens(text, max_tokens_per_request, encoding_name)
            text_tokens = len(encoding.encode(text))  # Recalculate after truncation
        
        # Check if adding this text would exceed limits
        would_exceed_tokens = current_batch_tokens + text_tokens > max_tokens_per_request
        would_exceed_items = (max_items_per_batch is not None and 
                             len(current_batch) >= max_items_per_batch)
        
        # If current batch is not empty and adding would exceed limits, yield current batch
        if current_batch and (would_exceed_tokens or would_exceed_items):
            yield current_batch
            current_batch = [text]
            current_batch_tokens = text_tokens
        else:
            current_batch.append(text)
            current_batch_tokens += text_tokens
    
    # Yield remaining batch
    if current_batch:
        yield current_batch


def get_batch_info(texts: Sequence[str], max_tokens_per_request: int = 8000000) -> dict:
    """
    Analyze texts and provide batch information.
    
    Returns:
        Dictionary with:
        - total_texts: Number of texts
        - total_tokens: Total token count
        - estimated_batches: Estimated number of batches
        - avg_tokens_per_text: Average tokens per text
        - max_tokens_in_single_text: Maximum tokens in any single text
        - recommended_batch_size: Recommended batch size
    """
    total_tokens = estimate_batch_tokens(texts)
    total_texts = len(texts)
    
    if total_texts == 0:
        return {
            "total_texts": 0,
            "total_tokens": 0,
            "estimated_batches": 0,
            "avg_tokens_per_text": 0,
            "max_tokens_in_single_text": 0,
            "recommended_batch_size": 0,
        }
    
    encoding = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(encoding.encode(text)) for text in texts]
    
    avg_tokens = total_tokens / total_texts
    max_tokens = max(token_counts)
    
    # Estimate batches (conservative estimate)
    estimated_batches = max(1, (total_tokens + max_tokens_per_request - 1) // max_tokens_per_request)
    
    # Recommend batch size (items, not tokens)
    if avg_tokens > 0:
        recommended_items_per_batch = min(200, max_tokens_per_request // int(avg_tokens * 1.2))
    else:
        recommended_items_per_batch = 100
    
    return {
        "total_texts": total_texts,
        "total_tokens": total_tokens,
        "estimated_batches": estimated_batches,
        "avg_tokens_per_text": avg_tokens,
        "max_tokens_in_single_text": max_tokens,
        "recommended_batch_size": recommended_items_per_batch,
    }


def call_embedding_api(
    texts: List[str],
    api_base: str,
    api_key: str,
    model: str,
    timeout: float = 120.0,
    rpm_limit: int = 0,
    sleep_between_requests: float = 0.0,
    use_openai_library: bool = True,
    dimensions: int | None = None,
) -> List[List[float]]:
    """
    Call OpenAI Embedding API to get embeddings for a batch of texts.
    
    Uses openai library by default (more reliable), falls back to requests if needed.
    
    Args:
        texts: List of text strings to embed
        api_base: API base URL (e.g., "https://api.openai.com/v1")
        api_key: API key
        model: Model name (e.g., "text-embedding-3-small")
        timeout: Request timeout in seconds
        rpm_limit: Requests per minute limit (0 = no limit)
        sleep_between_requests: Sleep time between requests in seconds
        use_openai_library: If True, use openai library; if False, use requests directly
        dimensions: Optional number of dimensions for output embeddings.
            Only supported in text-embedding-3 and later models.
            If None, uses model's default dimensions.
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    # Rate limiting: sleep if needed
    if sleep_between_requests > 0:
        time.sleep(sleep_between_requests)
    
    # Log request details
    print(f"  Making API request: {len(texts)} texts")
    
    # Use openai library if available (recommended)
    if use_openai_library and openai is not None:
        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
            )
            
            # Build parameters for embeddings.create
            create_params = {
                "model": model,
                "input": texts,
            }
            # Add dimensions only if specified and supported (text-embedding-3+)
            if dimensions is not None:
                create_params["dimensions"] = dimensions
            
            response = client.embeddings.create(**create_params)
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            # If openai library fails, fall back to requests
            if requests is None:
                raise RuntimeError(
                    f"OpenAI API call failed and requests library not available: {e}\n"
                    f"Please install requests: pip install requests"
                )
            # Fall through to requests method
    
    # Use requests if openai library not available or failed
    if requests is None:
        raise ImportError(
            "Either 'openai' or 'requests' library is required for OpenAI API. "
            "Install with: pip install openai (recommended) or pip install requests"
        )
    
    # Standard OpenAI API format: model in payload
    # Handle both cases: api_base with or without /v1
    api_base_clean = api_base.rstrip("/")
    if api_base_clean.endswith("/v1"):
        url = api_base_clean + "/embeddings"
    else:
        url = api_base_clean + "/v1/embeddings"
    payload = {
        "model": model,
        "input": texts,
    }
    # Add dimensions only if specified and supported (text-embedding-3+)
    if dimensions is not None:
        payload["dimensions"] = dimensions
    
    # Standard OpenAI API format - use Authorization: Bearer header
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Send request using json parameter (requests handles Content-Type automatically)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    
    if response.status_code != 200:
        error_text = response.text
        # Provide more helpful error messages
        if response.status_code == 401:
            error_msg = (
                f"Authentication failed (401): {error_text}\n"
                f"Please check:\n"
                f"  1. API key is correct and not expired\n"
                f"  2. API base URL is correct: {api_base}\n"
                f"  3. API key format is correct (should start with 'sk-')"
            )
        elif response.status_code == 429:
            error_msg = (
                f"Rate limit exceeded (429): {error_text}\n"
                f"Please wait and retry, or increase rpm_limit"
            )
        else:
            error_msg = f"API request failed ({response.status_code}): {error_text}"
        
        raise RuntimeError(error_msg)
    
    data = response.json()
    embeddings = [item["embedding"] for item in data.get("data", [])]
    
    return embeddings


class VectorRetriever:
    """
    Vector-based retriever supporting both local models and OpenAI API.
    
    Supports:
    - OpenAI API: text-embedding-3-small, text-embedding-3-large (with automatic batching)
    """
    
    def __init__(
        self,
        candidates: Sequence[Candidates],
        # Model parameters (choose one: API or local)
        api_base: str | None = None,  # Optional: API base URL (if using API)
        api_key: str | None = None,  # Optional: API key (if using API)
        model_name: str | None = None,  # Optional: API model name (if using API)
        # Local model parameters (alternative to API)
        local_model_name: str | None = None,  # Optional: Local SentenceTransformer model name (e.g., "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        normalize_embeddings: bool = True,
        max_tokens_per_request: int = 8192,  # Max total tokens per API request (only for API mode)
        max_items_per_batch: int | None = None,  # Optional: Max items per request (only for API mode, very short texts)
        rpm_limit: int = 300,  # Requests per minute limit (only for API mode, 0 = no limit)
        timeout: float = 120.0,  # Timeout for API requests (only for API mode)
        dimensions: int | None = None,  # Optional: Output dimensions (only for API mode, text-embedding-3+ only)
        # HNSW indexing parameters
        use_hnsw: bool = True,  # Whether to use HNSW index for faster search
        index_path: Path | str | None = None,  # Path to save/load HNSW index file or directory.
            # If None: uses default directory and auto-generates filename.
            # If directory: auto-generates filename based on model and dimensions.
            # If file path: uses the exact path.
        hnsw_m: int = 32,  # HNSW M parameter (number of connections per node, 16-64)
        hnsw_ef_construction: int = 100,  # HNSW ef_construction (build-time search range, 100-200)
        hnsw_ef_search: int = 64,  # HNSW ef_search (search-time search range, 16-256)
        # Embeddings caching parameters
        embeddings_dir: Path | str | None = None,  # Directory to save/load embeddings cache
        cache_embeddings: bool = True,  # Whether to cache embeddings to disk for reuse
        # Query embedding parameters (optional, defaults to local_model_name if not set)
        query_embedding_model: str | None = None,  # Optional: Local SentenceTransformer model for query embeddings (defaults to local_model_name if not set)
    ) -> None:
        self.candidates = list(candidates)
        
        # Determine if using API or local model
        if local_model_name:
            self.use_api = False
            self.local_model_name = local_model_name
            self.model_name = local_model_name  # For display/caching purposes
            # Get dimension from model (will be determined when model is loaded)
            self.dimensions = None  # Will be determined from model
            # Local model doesn't need API parameters
            self.api_base = None
            self.api_key = None
            self.max_tokens_per_request = None
            self.max_items_per_batch = None
            self.timeout = None
            self.sleep_between_requests = 0.0
        elif api_base or api_key:
            self.use_api = True
            self.local_model_name = None
            self.model_name = model_name
            self.dimensions = dimensions
            # API parameters
            if not api_key:
                raise ValueError("api_key is required when using API mode")
            if api_base is None:
                api_base = "https://api.openai.com/v1"
            self.api_base = api_base
            self.api_key = api_key
            self.max_tokens_per_request = max_tokens_per_request
            self.max_items_per_batch = max_items_per_batch
            self.timeout = timeout
            if rpm_limit > 0:
                self.sleep_between_requests = 60.0 / rpm_limit
            else:
                self.sleep_between_requests = 0.0
        else:
            raise ValueError("Either local_model_name or (api_base/api_key) must be provided")
        
        # Store query embedding model (defaults to local_model_name if not set)
        self.query_embedding_model = query_embedding_model or local_model_name
        self.query_model = None  # Will be initialized lazily if needed
        self.doc_model = None  # Will be initialized lazily if needed for local model
        
        # Pre-load query model if using local model for queries (to avoid delay on first query)
        if self.query_embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"Loading query embedding model: {self.query_embedding_model}")
            model_load_start = time.time()
            self.query_model = SentenceTransformer(self.query_embedding_model)
            model_load_time = time.time() - model_load_start
            print(f"✓ Query embedding model loaded in {model_load_time:.2f}s")
        
        # Determine index path: handle both file path and directory path
        if index_path is None:
            # Default: use artifacts/vector_indices directory
            self.index_dir = Path("artifacts") / "vector_indices"
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.index_path = None  # Will be auto-generated
        else:
            index_path_obj = Path(index_path)
            # Check if it's a file path (ends with .index or exists as file)
            if index_path_obj.suffix == ".index" or (index_path_obj.exists() and index_path_obj.is_file()):
                # It's a file path
                self.index_path = index_path_obj
                self.index_dir = index_path_obj.parent
            else:
                # It's a directory path
                self.index_dir = index_path_obj
                self.index_dir.mkdir(parents=True, exist_ok=True)
                self.index_path = None  # Will be auto-generated
        
        # Store embeddings caching parameters
        self.cache_embeddings = cache_embeddings
        if embeddings_dir is None:
            embeddings_dir = self.index_dir
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        
        # Store HNSW parameters
        self.use_hnsw = use_hnsw and hnswlib is not None
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.normalize_embeddings = normalize_embeddings
        
        # Determine if we can skip loading embeddings (if using HNSW index)
        # HNSW index contains vectors, so we only need embeddings if:
        # 1. Not using HNSW (fallback to linear search)
        # 2. Need to build a new index
        # 3. Need to verify index matches data
        
        # Try to load HNSW index first (if available and using HNSW)
        self.hnsw_index = None
        self.doc_embeddings = None
        embeddings_loaded = False
        index_loaded = False
        
        if self.use_hnsw:
            # Try to load existing index first
            # Use explicit path if provided, otherwise generate from directory
            if self.index_path is not None:
                index_path_obj = self.index_path
            else:
                # Use model_name for index path generation (works for both API and local models)
                index_path_obj = self._get_index_path(self.model_name, self.dimensions, len(candidates))
            
            if index_path_obj and index_path_obj.exists():
                try:
                    # Try to get dimension: use provided dimensions, or try to infer from embeddings cache
                    dimension = None
                    if self.dimensions:
                        dimension = self.dimensions
                    elif self.cache_embeddings and self.embeddings_dir:
                        # Try loading cached embeddings to get dimension
                        cached_embeddings = self._load_cached_embeddings(self.model_name, self.dimensions, len(candidates))
                        if cached_embeddings is not None:
                            dimension = cached_embeddings.shape[1]
                    
                    # If still no dimension, try to infer from model name
                    if dimension is None:
                        if self.local_model_name:
                            # For local models, try common dimensions based on model name
                            if 'minilm' in self.local_model_name.lower() and 'l12' in self.local_model_name.lower():
                                dimension = 384  # paraphrase-multilingual-MiniLM-L12-v2
                            elif 'multilingual' in self.local_model_name.lower():
                                dimension = 384  # Common for multilingual models
                            else:
                                dimension = 768  # Common default
                        elif self.model_name:
                            # Common dimensions for API embedding models
                            if 'small' in self.model_name.lower():
                                dimension = 1536  # text-embedding-3-small default
                            elif 'large' in self.model_name.lower():
                                dimension = 3072  # text-embedding-3-large default
                            else:
                                dimension = 768  # Common default
                        else:
                            dimension = 384  # Fallback default
                    
                    # Create index with estimated dimension
                    space = 'cosine' if self.normalize_embeddings else 'l2'
                    self.hnsw_index = hnswlib.Index(space=space, dim=dimension)
                    self.hnsw_index.load_index(str(index_path_obj))
                    
                    # Verify dimension matches
                    if self.hnsw_index.dim != dimension:
                        raise ValueError(
                            f"Index dimension mismatch: index has {self.hnsw_index.dim}, "
                            f"but expected {dimension}. Will rebuild index."
                        )
                    
                    self.hnsw_index.set_ef(self.hnsw_ef_search)
                    print(f"✓ Loaded HNSW index from: {index_path_obj.name}")
                    print(f"  Index contains {self.hnsw_index.element_count} vectors, dimension: {self.hnsw_index.dim}")
                    index_loaded = True
                    # If index is loaded, we don't need embeddings (index contains vectors)
                    # Only need candidates for mapping results back to documents
                    self.doc_embeddings = None
                    embeddings_loaded = True  # Mark as "loaded" to skip computation
                except Exception as e:
                    print(f"  ⚠ Failed to load index: {e}")
                    print(f"     Will rebuild index...")
                    self.hnsw_index = None
        
        # Only load embeddings if:
        # 1. Index not loaded AND we're using HNSW (need to build index)
        # 2. Not using HNSW (need for linear search)
        if not embeddings_loaded and not index_loaded:
            # Try to load cached embeddings first
            if self.cache_embeddings and self.embeddings_dir:
                cached_embeddings = self._load_cached_embeddings(self.model_name, self.dimensions, len(candidates))
                if cached_embeddings is not None:
                    self.doc_embeddings = cached_embeddings
                    # Update dimensions from cached embeddings
                    if self.dimensions is None:
                        self.dimensions = cached_embeddings.shape[1]
                    print(f"✓ Loaded cached embeddings. Shape: {self.doc_embeddings.shape}")
                    embeddings_loaded = True
        
        if not embeddings_loaded and not index_loaded:
            if self.use_api:
                # OpenAI API mode - compute embeddings using API
                # Note: API-related attributes are already set above
                self.normalize_embeddings = normalize_embeddings
                
                # Pre-compute document embeddings using API
                texts = [doc.text for doc in self.candidates]
                print(f"Computing embeddings for {len(texts)} documents using OpenAI API...")
                print(f"  Model: {self.model_name}")
                print(f"  Max tokens per request: {self.max_tokens_per_request:,}")
                if self.max_items_per_batch is not None:
                    print(f"  Max items per request: {self.max_items_per_batch}")
                else:
                    print(f"  Max items per request: unlimited (only token limit)")
                if self.dimensions is not None:
                    print(f"  Dimensions: {self.dimensions}")
                else:
                    print(f"  Dimensions: default (model-dependent)")
                
                # Get batch info
                batch_info = get_batch_info(texts, self.max_tokens_per_request)
                print(f"  Total tokens: {batch_info['total_tokens']:,}")
                print(f"  Estimated API requests: {batch_info['estimated_batches']}")
                
                # Batch and fetch embeddings
                all_embeddings = []
                batch_num = 0
                
                for batch_texts in batch_by_tokens(
                    texts,
                    max_tokens_per_request=self.max_tokens_per_request,
                    max_items_per_batch=self.max_items_per_batch,
                    model_name=self.model_name,
                ):
                    batch_num += 1
                    batch_tokens = estimate_batch_tokens(batch_texts)
                    print(f"  Processing batch {batch_num}: {len(batch_texts)} items, {batch_tokens:,} tokens")
                    
                    try:
                        batch_embeddings = call_embedding_api(
                            texts=batch_texts,
                            api_base=self.api_base,
                            api_key=self.api_key,
                            model=self.model_name,
                            timeout=self.timeout,
                            rpm_limit=rpm_limit,
                            sleep_between_requests=self.sleep_between_requests,
                            dimensions=self.dimensions,
                        )
                        all_embeddings.extend(batch_embeddings)
                    except Exception as e:
                        print(f"  Error processing batch {batch_num}: {e}")
                        raise
                
                self.doc_embeddings = np.array(all_embeddings)
                print(f"Embeddings computed. Shape: {self.doc_embeddings.shape}")
                
                # Update dimensions from computed embeddings
                if self.dimensions is None:
                    self.dimensions = self.doc_embeddings.shape[1]
                
                # Normalize if requested
                if normalize_embeddings:
                    norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                    self.doc_embeddings = self.doc_embeddings / norms
                
                # Save embeddings cache if enabled
                if self.cache_embeddings and self.embeddings_dir:
                    self._save_cached_embeddings(self.model_name, self.dimensions)
                
                # Mark embeddings as loaded
                embeddings_loaded = True
                
            elif self.local_model_name:
                # Local model mode - compute embeddings using SentenceTransformer
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError(
                        "SentenceTransformer is required for local model embeddings. "
                        "Install with: pip install sentence-transformers"
                    )
                
                # Initialize local model if not already initialized
                if self.doc_model is None:
                    print(f"Loading local embedding model: {self.local_model_name}")
                    model_load_start = time.time()
                    self.doc_model = SentenceTransformer(self.local_model_name)
                    model_load_time = time.time() - model_load_start
                    print(f"✓ Local model loaded in {model_load_time:.2f}s")
                    
                    # Get dimension from model
                    # Create a dummy embedding to get dimension
                    test_embedding = self.doc_model.encode(["test"], normalize_embeddings=normalize_embeddings)[0]
                    self.dimensions = len(test_embedding)
                    print(f"  Model dimension: {self.dimensions}")
                
                # Pre-compute document embeddings using local model
                texts = [doc.text for doc in self.candidates]
                print(f"Computing embeddings for {len(texts)} documents using local model...")
                print(f"  Model: {self.local_model_name}")
                print(f"  Dimension: {self.dimensions}")
                
                # Process in batches for better performance (local model can handle large batches)
                # Use a reasonable batch size for local processing
                batch_size = 32  # Process 32 items at a time
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(texts) + batch_size - 1) // batch_size
                    print(f"  Processing batch {batch_num}/{total_batches}: {len(batch_texts)} items")
                    
                    batch_embeddings = self.doc_model.encode(
                        batch_texts,
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=False
                    )
                    all_embeddings.append(batch_embeddings)
                
                # Concatenate all batches
                self.doc_embeddings = np.vstack(all_embeddings)
                print(f"Embeddings computed. Shape: {self.doc_embeddings.shape}")
                
                # Save embeddings cache if enabled
                if self.cache_embeddings and self.embeddings_dir:
                    self._save_cached_embeddings(self.model_name, self.dimensions)
                
                # Mark embeddings as loaded
                embeddings_loaded = True
                
            else:
                # This should never happen
                raise ValueError(
                    "Either use_api=True or local_model_name must be provided. "
                    "This error should not occur."
                )
        
        # Build HNSW index if needed (and not already loaded)
        if self.use_hnsw and not index_loaded:
            if self.doc_embeddings is not None:
                # We have embeddings, build index
                self._build_hnsw_index()
            else:
                print("  ⚠ Cannot build HNSW index: embeddings not available")
                print("     Falling back to linear search.")
                self.use_hnsw = False
        elif not self.use_hnsw:
            if use_hnsw and hnswlib is None:
                print("  ⚠ HNSW library not available. Install with: pip install hnswlib")
                print("     Falling back to linear search.")
    
    def _get_embeddings_cache_path(self, model_name: str, dimensions: int | None, num_docs: int) -> Path | None:
        """Generate embeddings cache file path."""
        if not self.embeddings_dir:
            return None
        
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        dim_str = str(dimensions) if dimensions else "default"
        model_str = model_name.replace("/", "_").replace("-", "_")
        cache_filename = f"embeddings_{model_str}_dim{dim_str}_{num_docs}.npy"
        return self.embeddings_dir / cache_filename
    
    def _load_cached_embeddings(self, model_name: str, dimensions: int | None, num_docs: int) -> np.ndarray | None:
        """Load cached embeddings from disk if available."""
        cache_path = self._get_embeddings_cache_path(model_name, dimensions, num_docs)
        if cache_path and cache_path.exists():
            try:
                print(f"  Loading cached embeddings from: {cache_path.name}")
                embeddings = np.load(str(cache_path))
                print(f"  ✓ Cached embeddings loaded successfully")
                return embeddings
            except Exception as e:
                print(f"  ⚠ Failed to load cached embeddings: {e}")
                print(f"     Will compute new embeddings.")
        return None
    
    def _save_cached_embeddings(self, model_name: str, dimensions: int | None) -> None:
        """Save embeddings to disk cache."""
        if not hasattr(self, 'doc_embeddings') or self.doc_embeddings is None:
            return
        
        cache_path = self._get_embeddings_cache_path(model_name, dimensions, len(self.candidates))
        if cache_path:
            try:
                np.save(str(cache_path), self.doc_embeddings)
                cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Saved embeddings cache: {cache_path.name} ({cache_size_mb:.2f} MB)")
            except Exception as e:
                print(f"  ⚠ Failed to save embeddings cache: {e}")
    
    def _get_index_path(self, model_name: str, dimensions: int | None, num_docs: int) -> Path | None:
        """Generate HNSW index file path."""
        if not self.index_dir:
            return None
        
        dim_str = str(dimensions) if dimensions else "default"
        model_str = model_name.replace("/", "_").replace("-", "_")
        index_filename = f"hnsw_{model_str}_dim{dim_str}_{num_docs}.index"
        return self.index_dir / index_filename
    
    def _build_hnsw_index(self) -> None:
        """Build HNSW index from embeddings and save to disk."""
        if self.doc_embeddings is None:
            raise ValueError("Cannot build index: embeddings not available")
        
        dimension = self.doc_embeddings.shape[1]
        num_docs = len(self.doc_embeddings)
        
        print(f"  Building HNSW index...")
        print(f"    Dimension: {dimension}")
        print(f"    Documents: {num_docs}")
        print(f"    M: {self.hnsw_m}, ef_construction: {self.hnsw_ef_construction}, ef_search: {self.hnsw_ef_search}")
        
        # Convert embeddings to float32 (required by hnswlib)
        embeddings_f32 = self.doc_embeddings.astype('float32')
        
        # Normalize embeddings if needed (hnswlib 'cosine' space requires normalized vectors)
        if self.normalize_embeddings:
            # Normalize each vector
            norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings_f32 = embeddings_f32 / norms
        
        # Create HNSW index
        # Use 'cosine' space if embeddings are normalized, 'l2' otherwise
        space = 'cosine' if self.normalize_embeddings else 'l2'
        self.hnsw_index = hnswlib.Index(space=space, dim=dimension)
        
        # Initialize index with parameters
        self.hnsw_index.init_index(max_elements=num_docs, M=self.hnsw_m, ef_construction=self.hnsw_ef_construction)
        
        # Add vectors to index (with IDs 0, 1, 2, ...)
        self.hnsw_index.add_items(embeddings_f32, np.arange(num_docs))
        
        # Set ef_search for query time
        self.hnsw_index.set_ef(self.hnsw_ef_search)
        
        print(f"  ✓ HNSW index built successfully")
        
        # Save index
        # Use explicit path if provided, otherwise generate from directory
        if self.index_path is not None:
            save_path = self.index_path
        else:
            save_path = self._get_index_path(self.model_name, self.dimensions, num_docs)
        
        if save_path:
            try:
                self.hnsw_index.save_index(str(save_path))
                print(f"  ✓ HNSW index saved to: {save_path.name}")
            except Exception as e:
                print(f"  ⚠ Failed to save index: {e}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Candidates, float]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (candidate, score) tuples, sorted by relevance score (highest first)
        """
        if not query.strip():
            return []
        
        # Encode query
        # Use local model for query embedding if specified, otherwise use API
        if self.query_embedding_model:
            # Use local SentenceTransformer model for query embedding
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "SentenceTransformer is required for local query embedding. "
                    "Install with: pip install sentence-transformers"
                )
            
            # Query model should already be loaded in __init__, but check just in case
            if self.query_model is None:
                # Fallback: lazy initialization if somehow not loaded
                print(f"  Loading local query embedding model: {self.query_embedding_model}")
                query_embedding_start = time.time()
                self.query_model = SentenceTransformer(self.query_embedding_model)
                load_time = time.time() - query_embedding_start
                print(f"  ✓ Local query model loaded in {load_time:.2f}s")
            
            query_embedding_start = time.time()
            query_embedding = self.query_model.encode([query], normalize_embeddings=self.normalize_embeddings)[0]
            query_embedding_time = time.time() - query_embedding_start
            print(f"  Query embedding (local model) completed in {query_embedding_time:.3f}s")
            query_embedding = np.array(query_embedding)
            
            # Check dimension mismatch (query embedding vs document embeddings) - CRITICAL
            query_dim = len(query_embedding)
            if self.use_hnsw and self.hnsw_index is not None:
                doc_dim = self.hnsw_index.dim
                if query_dim != doc_dim:
                    raise ValueError(
                        f"Dimension mismatch: Query embedding dimension ({query_dim}) != Document embedding dimension ({doc_dim}).\n"
                        f"  Query model: {self.query_embedding_model}\n"
                        f"  Document model: {self.model_name}\n"
                        f"  This will cause search to fail. Please use the same embedding model for both queries and documents,\n"
                        f"  or ensure the dimensions match."
                    )
            elif self.doc_embeddings is not None:
                doc_dim = self.doc_embeddings.shape[1]
                if query_dim != doc_dim:
                    raise ValueError(
                        f"Dimension mismatch: Query embedding dimension ({query_dim}) != Document embedding dimension ({doc_dim}).\n"
                        f"  Query model: {self.query_embedding_model}\n"
                        f"  Document model: {self.model_name}\n"
                        f"  This will cause search to fail. Please use the same embedding model for both queries and documents,\n"
                        f"  or ensure the dimensions match."
                    )
        elif self.use_api:
            # Use API for query embedding
            query_embedding_start = time.time()
            query_embeddings = call_embedding_api(
                texts=[query],
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.model_name,
                timeout=self.timeout,
                rpm_limit=0,  # Don't sleep for single query
                sleep_between_requests=0.0,
                dimensions=self.dimensions,
            )
            query_embedding_time = time.time() - query_embedding_start
            print(f"  Query embedding API call completed in {query_embedding_time:.2f}s")
            query_embedding = np.array(query_embeddings[0])
            
            # Normalize if requested
            if self.normalize_embeddings:
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
        else:
            # This should never happen - API mode is required for document embeddings
            raise ValueError("API mode is required for document embeddings. Query embedding can use local model if query_embedding_model is specified.")
        
        # Use HNSW index if available, otherwise use linear search
        if self.use_hnsw and self.hnsw_index is not None:
            # Use HNSW index for fast search
            query_vec = query_embedding.astype('float32')
            
            # Normalize query vector if using normalized embeddings (for cosine similarity)
            if self.normalize_embeddings:
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm
            
            # Search using HNSW
            hnsw_search_start = time.time()
            labels, distances = self.hnsw_index.knn_query(query_vec, k=top_k)
            hnsw_search_time = time.time() - hnsw_search_start
            print(f"  HNSW search completed in {hnsw_search_time:.3f}s")
            
            # Convert distances to similarity scores
            # For cosine similarity: distance = 1 - similarity, so similarity = 1 - distance
            # For L2 distance: convert to similarity using 1 / (1 + distance)
            results: List[Tuple[Candidates, float]] = []
            for idx, dist in zip(labels[0], distances[0]):
                if idx < 0 or idx >= len(self.candidates):
                    continue  # Skip invalid indices
                
                if self.normalize_embeddings:
                    # Cosine distance: convert to similarity (1 - distance)
                    # Note: hnswlib returns cosine distance (0 = same, 2 = opposite)
                    score = float(1.0 - dist)  # Convert distance to similarity
                else:
                    # Convert L2 distance to similarity (higher is better)
                    # Use 1 / (1 + distance) to ensure score is in [0, 1] range
                    score = float(1.0 / (1.0 + dist))
                
                if score > 0:  # Only include positive similarity scores
                    results.append((self.candidates[idx], score))
            
            return results
        else:
            # Fallback to linear search
            # Compute similarity scores (cosine similarity if normalized, dot product otherwise)
            if self.normalize_embeddings:
                # Cosine similarity = dot product when embeddings are normalized
                scores = np.dot(self.doc_embeddings, query_embedding)
            else:
                # Use cosine similarity manually
                query_norm = np.linalg.norm(query_embedding)
                doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)
                scores = np.dot(self.doc_embeddings, query_embedding) / (doc_norms * query_norm)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Filter out negative scores (or low similarity)
            results: List[Tuple[Candidates, float]] = []
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0:  # Only include positive similarity scores
                    results.append((self.candidates[idx], score))
            
            return results


def load_food_candidates_for_vector(csv_path: Path) -> List[Candidates]:
    return load_food_candidates(csv_path)


# ---------------------------------------------------------------------------
# Token calculation examples and utilities
# ---------------------------------------------------------------------------


def demonstrate_sample_query(
    data_dir: Path,
    use_openai_api: bool,
    api_base: str,
    api_key: str,
    dimensions: int,
    index_path: Path | str,
    model_name: str,
    max_tokens_per_request: int,
    max_items_per_batch: int | None,
    rpm_limit: int,
    timeout: float,
    normalize_embeddings: bool,
    use_hnsw: bool,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    embeddings_dir: Path | str | None,
    cache_embeddings: bool,
) -> None:

    items_path = data_dir / "500_items.csv"
    queries_path = data_dir / "10_queries.csv"

    print("=" * 80)
    print("Vector Retrieval with HNSW Index Demo")
    print("=" * 80)
    
    # Validate all required parameters - no defaults allowed
    if not use_openai_api:
        raise ValueError("use_openai_api must be True. Local model mode is not supported.")
    
    if not api_key:
        raise ValueError("api_key is required and must be provided explicitly.")
    
    if not api_base:
        raise ValueError("api_base is required and must be provided explicitly (e.g., 'https://api.openai.com/v1').")
    
    if index_path is None:
        raise ValueError("index_path is required and must be provided explicitly.")
    
    if not model_name:
        raise ValueError("model_name is required and must be provided explicitly (e.g., 'text-embedding-3-small').")
    
    if dimensions is None or dimensions <= 0:
        raise ValueError("dimensions is required and must be a positive integer (e.g., 1536 for text-embedding-3-small).")
    
    print("\n1. Loading candidates...")
    candidates = load_food_candidates_for_vector(items_path)
    print(f"   ✓ Loaded {len(candidates)} candidates")
    
    print("\n2. Index path: ", end="")
    index_path_obj = Path(index_path)
    if index_path_obj.is_dir():
        index_path_obj.mkdir(parents=True, exist_ok=True)
        # Generate expected index filename
        dim_str = str(dimensions)
        num_docs = len(candidates)
        expected_index_name = f"hnsw_{model_name.replace('/', '_').replace('-', '_')}_dim{dim_str}_{num_docs}.index"
        expected_index_path = index_path_obj / expected_index_name
        print(f"{index_path_obj} (will use: {expected_index_name})")
    else:
        # It's a file path
        expected_index_path = index_path_obj
        print(f"{index_path_obj}")
    
    if expected_index_path.exists():
        print(f"   ✓ Found existing index: {expected_index_path.name}")
        print(f"     This index will be automatically loaded!")
    else:
        print(f"   ⚠ No existing index found. Will build new index and save it.")
    
    print("\n3. Initializing vector retriever...")
    print("   (This will compute embeddings and build/load HNSW index)")
    print(f"   Using API: {api_base}")
    print(f"   Model: {model_name}")
    print(f"   Dimensions: {dimensions}")
    
    # Determine embeddings directory
    if embeddings_dir is None:
        if index_path_obj.is_file():
            embeddings_dir = index_path_obj.parent
        else:
            embeddings_dir = index_path_obj
    
    retriever = VectorRetriever(
        candidates,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        max_tokens_per_request=max_tokens_per_request,
        max_items_per_batch=max_items_per_batch,
        rpm_limit=rpm_limit,
        timeout=timeout,
        dimensions=dimensions,
        normalize_embeddings=normalize_embeddings,
        # HNSW indexing parameters
        use_hnsw=use_hnsw,
        index_path=index_path_obj,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        # Embeddings caching parameters
        embeddings_dir=Path(embeddings_dir) if embeddings_dir else None,
        cache_embeddings=cache_embeddings,
    )

    print("\n4. Performing search...")
    with queries_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

    query = first_row["search_term_pt"]
    print(f"   Query: {query}")
    
    import time
    start_time = time.time()
    results = retriever.search(query, top_k=5)
    search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    print(f"\n5. Search Results (search time: {search_time:.2f}ms):")
    print("-" * 80)
    for rank, (doc, score) in enumerate(results, start=1):
        name = doc.name[:60] + "..." if len(doc.name) > 60 else doc.name
        print(f"   {rank}. score={score:.4f} | itemId={doc.id}")
        print(f"      name: {name}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    # Get actual index path (either explicit or generated)
    if retriever.index_path is not None:
        actual_index_path = retriever.index_path
    else:
        actual_index_path = retriever._get_index_path(model_name, dimensions, len(candidates))
    
    if actual_index_path:
        print(f"  • Index location: {actual_index_path}")
        if actual_index_path.exists():
            index_size_mb = actual_index_path.stat().st_size / (1024 * 1024)
            print(f"  • Index size: {index_size_mb:.2f} MB")
    
    # Show embeddings cache info
    if retriever.embeddings_dir:
        cache_path = retriever._get_embeddings_cache_path(
            model_name, dimensions, len(candidates)
        )
        if cache_path and cache_path.exists():
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"  • Embeddings cache: {cache_path.name} ({cache_size_mb:.2f} MB)")
    
    print(f"  • Search method: {'HNSW (fast)' if retriever.use_hnsw else 'Linear search (slow)'}")
    print(f"  • Search time: {search_time:.2f}ms")
    print(f"\n💡 Tip: Next time you run this script:")
    print(f"   • Embeddings will be loaded from cache (no API calls needed!)")
    print(f"   • HNSW index will be loaded automatically")
    print(f"   • Only query embeddings will be computed - super fast! 🚀")
    print("=" * 80)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    
    # All parameters must be explicitly set - no defaults or auto-detection
    print("=" * 80)
    print("HNSW Vector Database Demo")
    print("=" * 80)
    print("\n⚠️  All parameters must be explicitly provided. No defaults will be used.")
    print("=" * 80 + "\n")
    
    # Get API key from .env file or environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required. Set it in .env file or environment variable."
        )
    
    # Get API base from .env file or environment variable
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_base or not api_base.strip():
        raise ValueError(
            "OPENAI_API_BASE is required. Set it in .env file or environment variable (e.g., 'https://api.openai.com/v1')."
        )
    api_base = api_base.strip()
    
    # Get model name from environment variable
    model_name = os.getenv("OPENAI_EMBEDDING_MODEL")
    if not model_name or not model_name.strip():
        raise ValueError(
            "OPENAI_EMBEDDING_MODEL is required. Set it in .env file or environment variable (e.g., 'text-embedding-3-small')."
        )
    model_name = model_name.strip()
    
    # Get dimensions from environment variable
    dimensions_str = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
    if not dimensions_str or not dimensions_str.strip():
        raise ValueError(
            "OPENAI_EMBEDDING_DIMENSIONS is required. Set it in .env file or environment variable (e.g., '1536' for text-embedding-3-small)."
        )
    try:
        dimensions = int(dimensions_str.strip())
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
    except ValueError as e:
        raise ValueError(f"Invalid OPENAI_EMBEDDING_DIMENSIONS value '{dimensions_str}': {e}")
    
    # Get index path from environment variable
    index_path_str = os.getenv("VECTOR_INDEX_PATH")
    if not index_path_str or not index_path_str.strip():
        raise ValueError(
            "VECTOR_INDEX_PATH is required. Set it in .env file or environment variable (e.g., 'artifacts/vector_indices')."
        )
    index_path = project_root / index_path_str.strip()
    
    print("\nConfiguration:")
    print(f"  • API Base: {api_base}")
    print(f"  • Model: {model_name}")
    print(f"  • Dimensions: {dimensions}")
    print(f"  • Index Path: {index_path}")
    print("=" * 80 + "\n")
    
    # Get all other required parameters from environment variables
    max_tokens_per_request_str = os.getenv("MAX_TOKENS_PER_REQUEST")
    if not max_tokens_per_request_str or not max_tokens_per_request_str.strip():
        raise ValueError(
            "MAX_TOKENS_PER_REQUEST is required. Set it in .env file or environment variable (e.g., '8192')."
        )
    try:
        max_tokens_per_request = int(max_tokens_per_request_str.strip())
        if max_tokens_per_request <= 0:
            raise ValueError(f"max_tokens_per_request must be positive, got {max_tokens_per_request}")
    except ValueError as e:
        raise ValueError(f"Invalid MAX_TOKENS_PER_REQUEST value '{max_tokens_per_request_str}': {e}")
    
    max_items_per_batch_str = os.getenv("MAX_ITEMS_PER_BATCH")
    max_items_per_batch = None
    if max_items_per_batch_str and max_items_per_batch_str.strip():
        try:
            max_items_per_batch = int(max_items_per_batch_str.strip())
            if max_items_per_batch <= 0:
                raise ValueError(f"max_items_per_batch must be positive, got {max_items_per_batch}")
        except ValueError as e:
            raise ValueError(f"Invalid MAX_ITEMS_PER_BATCH value '{max_items_per_batch_str}': {e}")
    
    rpm_limit_str = os.getenv("RPM_LIMIT")
    if not rpm_limit_str or not rpm_limit_str.strip():
        raise ValueError(
            "RPM_LIMIT is required. Set it in .env file or environment variable (e.g., '300')."
        )
    try:
        rpm_limit = int(rpm_limit_str.strip())
        if rpm_limit <= 0:
            raise ValueError(f"rpm_limit must be positive, got {rpm_limit}")
    except ValueError as e:
        raise ValueError(f"Invalid RPM_LIMIT value '{rpm_limit_str}': {e}")
    
    timeout_str = os.getenv("TIMEOUT")
    if not timeout_str or not timeout_str.strip():
        raise ValueError(
            "TIMEOUT is required. Set it in .env file or environment variable (e.g., '120.0')."
        )
    try:
        timeout = float(timeout_str.strip())
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
    except ValueError as e:
        raise ValueError(f"Invalid TIMEOUT value '{timeout_str}': {e}")
    
    normalize_embeddings_str = os.getenv("NORMALIZE_EMBEDDINGS", "True").strip().lower()
    normalize_embeddings = normalize_embeddings_str in ("true", "1", "yes")
    
    use_hnsw_str = os.getenv("USE_HNSW", "True").strip().lower()
    use_hnsw = use_hnsw_str in ("true", "1", "yes")
    
    hnsw_m_str = os.getenv("HNSW_M")
    if not hnsw_m_str or not hnsw_m_str.strip():
        raise ValueError(
            "HNSW_M is required. Set it in .env file or environment variable (e.g., '32')."
        )
    try:
        hnsw_m = int(hnsw_m_str.strip())
        if hnsw_m <= 0:
            raise ValueError(f"hnsw_m must be positive, got {hnsw_m}")
    except ValueError as e:
        raise ValueError(f"Invalid HNSW_M value '{hnsw_m_str}': {e}")
    
    hnsw_ef_construction_str = os.getenv("HNSW_EF_CONSTRUCTION")
    if not hnsw_ef_construction_str or not hnsw_ef_construction_str.strip():
        raise ValueError(
            "HNSW_EF_CONSTRUCTION is required. Set it in .env file or environment variable (e.g., '100')."
        )
    try:
        hnsw_ef_construction = int(hnsw_ef_construction_str.strip())
        if hnsw_ef_construction <= 0:
            raise ValueError(f"hnsw_ef_construction must be positive, got {hnsw_ef_construction}")
    except ValueError as e:
        raise ValueError(f"Invalid HNSW_EF_CONSTRUCTION value '{hnsw_ef_construction_str}': {e}")
    
    hnsw_ef_search_str = os.getenv("HNSW_EF_SEARCH")
    if not hnsw_ef_search_str or not hnsw_ef_search_str.strip():
        raise ValueError(
            "HNSW_EF_SEARCH is required. Set it in .env file or environment variable (e.g., '64')."
        )
    try:
        hnsw_ef_search = int(hnsw_ef_search_str.strip())
        if hnsw_ef_search <= 0:
            raise ValueError(f"hnsw_ef_search must be positive, got {hnsw_ef_search}")
    except ValueError as e:
        raise ValueError(f"Invalid HNSW_EF_SEARCH value '{hnsw_ef_search_str}': {e}")
    
    embeddings_dir_str = os.getenv("EMBEDDINGS_DIR")
    embeddings_dir = None
    if embeddings_dir_str and embeddings_dir_str.strip():
        embeddings_dir = project_root / embeddings_dir_str.strip()
    
    cache_embeddings_str = os.getenv("CACHE_EMBEDDINGS", "True").strip().lower()
    cache_embeddings = cache_embeddings_str in ("true", "1", "yes")
    
    # Call demonstrate_sample_query with all required parameters
    demonstrate_sample_query(
        data_dir=project_root / "data" / "test",
        use_openai_api=True,
        api_base=api_base,
        api_key=api_key,
        dimensions=dimensions,
        index_path=index_path,
        model_name=model_name,
        max_tokens_per_request=max_tokens_per_request,
        max_items_per_batch=max_items_per_batch,
        rpm_limit=rpm_limit,
        timeout=timeout,
        normalize_embeddings=normalize_embeddings,
        use_hnsw=use_hnsw,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        embeddings_dir=embeddings_dir,
        cache_embeddings=cache_embeddings,
    )

