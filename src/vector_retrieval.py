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
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # Optional for OpenAI API mode

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
    - Local models: SentenceTransformer (offline, no API calls)
    - OpenAI API: text-embedding-3-small, text-embedding-3-large (with automatic batching)
    """
    
    def __init__(
        self,
        candidates: Sequence[Candidates],
        # Local model parameters
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None,
        normalize_embeddings: bool = True,
        # OpenAI API parameters
        api_base: str | None = None,
        api_key: str | None = None,
        max_tokens_per_request: int = 8192,  # Max total tokens per API request
        max_items_per_batch: int | None = None,  # Optional: Max items per request (only for very short texts)
        rpm_limit: int = 300,  # Requests per minute limit (0 = no limit)
        timeout: float = 120.0,
        dimensions: int | None = None,  # Optional: Output dimensions (text-embedding-3+ only)
    ) -> None:
        """
        Initialize vector retriever.
        
        Args:
            candidates: Sequence of candidate documents
            # Local model parameters
            model_name: Name of the sentence transformer model (if using local)
            device: Device to use ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to normalize embeddings for cosine similarity
            # OpenAI API parameters
            api_base: API base URL (if None, uses local model)
            api_key: API key (required if api_base is set)
            max_tokens_per_request: Maximum total tokens per API request (for automatic batching)
            max_items_per_batch: Optional maximum items per request. Only relevant when texts
                are very short. If None, only token limit is used. Default None.
            rpm_limit: Requests per minute limit (auto-calculates sleep if set)
            timeout: Request timeout in seconds
            dimensions: Optional number of dimensions for output embeddings.
                Only supported in text-embedding-3 and later models.
                If None, uses model's default dimensions.
        """
        self.candidates = list(candidates)
        
        # Determine if using API: either api_base is provided, or api_key is provided (use standard OpenAI API)
        self.use_api = api_base is not None or (api_key is not None and api_key)
        
        if self.use_api:
            # OpenAI API mode
            if not api_key:
                raise ValueError("api_key is required when using OpenAI API")
            
            # If api_base is None but api_key is provided, use standard OpenAI API
            if api_base is None:
                api_base = "https://api.openai.com/v1"
            
            self.api_base = api_base
            self.api_key = api_key
            self.model_name = model_name  # e.g., "text-embedding-3-small"
            self.max_tokens_per_request = max_tokens_per_request
            self.max_items_per_batch = max_items_per_batch
            self.timeout = timeout
            self.dimensions = dimensions
            
            # Calculate sleep time based on RPM limit
            if rpm_limit > 0:
                self.sleep_between_requests = 60.0 / rpm_limit
            else:
                self.sleep_between_requests = 0.0
            
            self.normalize_embeddings = normalize_embeddings
            
            # Pre-compute document embeddings using API
            texts = [doc.text for doc in self.candidates]
            print(f"Computing embeddings for {len(texts)} documents using OpenAI API...")
            print(f"  Model: {model_name}")
            print(f"  Max tokens per request: {max_tokens_per_request:,}")
            if max_items_per_batch is not None:
                print(f"  Max items per request: {max_items_per_batch}")
            else:
                print(f"  Max items per request: unlimited (only token limit)")
            if dimensions is not None:
                print(f"  Dimensions: {dimensions}")
            else:
                print(f"  Dimensions: default (model-dependent)")
            
            # Get batch info
            batch_info = get_batch_info(texts, max_tokens_per_request)
            print(f"  Total tokens: {batch_info['total_tokens']:,}")
            print(f"  Estimated API requests: {batch_info['estimated_batches']}")
            
            # Batch and fetch embeddings
            all_embeddings = []
            batch_num = 0
            
            for batch_texts in batch_by_tokens(
                texts,
                max_tokens_per_request=max_tokens_per_request,
                max_items_per_batch=max_items_per_batch,
                model_name=model_name,
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
            
            # Normalize if requested
            if normalize_embeddings:
                norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                self.doc_embeddings = self.doc_embeddings / norms
        else:
            # Local model mode
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence_transformers is required for local model mode. "
                    "Install it with: pip install sentence-transformers"
                )
            
            self.model = SentenceTransformer(model_name, device=device)
            self.normalize_embeddings = normalize_embeddings
            
            # Pre-compute document embeddings
            texts = [doc.text for doc in self.candidates]
            print(f"Computing embeddings for {len(texts)} documents using local model...")
            self.doc_embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=normalize_embeddings,
            )
            # Convert to numpy array for efficient computation
            self.doc_embeddings = np.array(self.doc_embeddings)
            print(f"Embeddings computed. Shape: {self.doc_embeddings.shape}")
    
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
        if self.use_api:
            # Use API for query embedding
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
            query_embedding = np.array(query_embeddings[0])
            
            # Normalize if requested
            if self.normalize_embeddings:
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
        else:
            # Use local model
            query_embedding = self.model.encode(
                query,
                normalize_embeddings=self.normalize_embeddings,
            )
            query_embedding = np.array(query_embedding)
        
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
    use_openai_api: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    dimensions: int | None = None,
) -> None:
    """
    Demonstrate vector retrieval with a sample query.
    
    Args:
        data_dir: Directory containing data files
        use_openai_api: If True, use OpenAI API; if False, use local SentenceTransformer model
        api_base: API base URL (required if use_openai_api=True)
        api_key: API key (required if use_openai_api=True)
        dimensions: Optional number of dimensions for output embeddings.
            Only supported in text-embedding-3 and later models.
            If None, uses model's default dimensions.
    """
    items_path = data_dir / "500_items.csv"
    queries_path = data_dir / "10_queries.csv"

    print("Loading candidates...")
    candidates = load_food_candidates_for_vector(items_path)
    
    print("Initializing vector retriever...")
    if use_openai_api:
        if not api_key:
            raise ValueError("api_key is required when use_openai_api=True")
        
        # Use standard OpenAI API if api_base not provided
        if not api_base:
            api_base = "https://api.openai.com/v1"
            print(f"  Using standard OpenAI API: {api_base}")
        else:
            print(f"  Using custom API: {api_base}")
        
        print(f"  Model: text-embedding-3-small")
        if dimensions is not None:
            print(f"  Dimensions: {dimensions}")
        else:
            print(f"  Dimensions: default (model-dependent)")
        # text-embedding-3-small supports up to 8191 tokens per input
        # For batch requests, use a conservative limit
        retriever = VectorRetriever(
            candidates,
            api_base=api_base,
            api_key=api_key,
            model_name="text-embedding-3-small",
            max_tokens_per_request=8192,  # Conservative limit for batch processing
            max_items_per_batch=100,  # Limit items per batch to avoid server issues
            rpm_limit=500,  # OpenAI default is higher, but use conservative limit
            dimensions=dimensions,  # Pass dimensions parameter
        )
    else:
        print("  Using local SentenceTransformer model")
        retriever = VectorRetriever(
            candidates,
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

    with queries_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

    query = first_row["search_term_pt"]
    print(f"\nQuery: {query}")
    
    results = retriever.search(query, top_k=5)
    
    print("\nTop-5 results:")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}. score={score:.4f} itemId={doc.id} name={doc.name}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    
    # Get API key from .env file or environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("=" * 70)
        print("ERROR: OPENAI_API_KEY not found!")
        print("=" * 70)
        print("Please set your OpenAI API key in one of the following ways:")
        print("\n1. Create a .env file in the project root with:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        print("\n2. Or set environment variable:")
        print("   export OPENAI_API_KEY='sk-your-key-here'")
        print("\n3. Install python-dotenv if you want to use .env file:")
        print("   pip install python-dotenv")
        print("=" * 70)
        exit(1)
    
    # Get optional API base from .env file or environment variable
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base and api_base.strip():
        api_base = api_base.strip()
    else:
        api_base = None  # None means use standard OpenAI API
    
    # Get optional dimensions from .env file or environment variable
    dimensions_str = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
    dimensions = None
    if dimensions_str and dimensions_str.strip():
        try:
            dimensions = int(dimensions_str.strip())
        except ValueError:
            print(f"Warning: Invalid OPENAI_EMBEDDING_DIMENSIONS value '{dimensions_str}', ignoring.")
    
    # Use standard OpenAI API
    demonstrate_sample_query(
        project_root / "data" / "test",
        use_openai_api=True,
        api_base=api_base,  # None means use standard OpenAI API
        api_key=api_key,
        dimensions=dimensions,  # Pass dimensions if specified
    )

