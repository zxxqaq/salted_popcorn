from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(
    r"[0-9]+(?:\.[0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ0-9']+"
)

def tokenize(text: str) -> List[str]:
    """
    Tokenize text by splitting on underscores and other delimiters.
    
    Replaces underscores with spaces so they act as word separators.
    This ensures that if any text contains underscores (e.g., "não_vegano"),
    it will be tokenized as ["não", "vegano"] instead of ["não_vegano"],
    allowing queries like "não vegano" to match.
    
    Note: build_candidate_text() now generates text with spaces instead of underscores,
    so this function primarily handles edge cases or other text sources that may contain underscores.
    """
    # Replace underscores with spaces so they act as word separators
    text_normalized = text.replace("_", " ")
    tokens = TOKEN_PATTERN.findall(text_normalized.lower())
    return tokens


def build_candidate_text(metadata: dict) -> str:
    pieces: List[str] = []

    name = metadata.get("name")
    if name:
        pieces.append(str(name))

    category = metadata.get("category_name")
    if category:
        pieces.append(str(category))

    description = metadata.get("description")
    if description:
        pieces.append(str(description))

    taxonomy = metadata.get("taxonomy", {})
    taxonomy_values: Iterable[str]
    if isinstance(taxonomy, dict):
        taxonomy_values = taxonomy.values()
    elif isinstance(taxonomy, Sequence) and not isinstance(taxonomy, (str, bytes)):
        taxonomy_values = taxonomy
    else:
        taxonomy_values = ()
    for value in taxonomy_values:
        if value:
            pieces.append(str(value))

    tags = metadata.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, dict):
                continue
            value = tag.get("value")
            values: Iterable[str]
            if isinstance(value, list):
                values = value
            elif value is None:
                values = ()
            else:
                values = (value,)
            for item in values:
                if not item:
                    continue
                if isinstance(item, str) and item.upper() == "NOT_APPLICABLE":
                    continue
                pieces.append(str(item))

    price = metadata.get("price")
    if price is not None:
        pieces.append(str(price))

    if metadata.get("lacFree") is True:
        pieces.append("sem lactose")
    elif metadata.get("lacFree") is False:
        pieces.append("com lactose")

    if metadata.get("organic") is True:
        pieces.append("orgânico")
    elif metadata.get("organic") is False:
        pieces.append("não orgânico")

    if metadata.get("vegan") is True:
        pieces.append("vegano")
    elif metadata.get("vegan") is False:
        pieces.append("não vegano")

    return " ".join(pieces)


@dataclass
class Candidates:
    id: str
    name: str
    text: str
    tokens: List[str]


class BM25Retriever:
    """
    BM25 retriever with caching support.
    
    Supports persisting the entire BM25 index (including tokens, df/idf, avgdl, etc.)
    to disk and loading it on startup for faster cold start.
    """
    
    def __init__(
        self,
        candidate: Sequence[Candidates],
        k1: float = 1.5,
        b: float = 0.75,
        cache_dir: Path | str | None = None,
        cache_enabled: bool = True,
        data_source_hash: str | None = None,
    ) -> None:
        """
        Initialize BM25 retriever.
        
        Args:
            candidate: Sequence of candidate documents
            k1: BM25 k1 parameter
            b: BM25 b parameter
            cache_dir: Directory to save/load BM25 cache (None = use default)
            cache_enabled: Whether to enable caching
            data_source_hash: Optional hash of data source for cache validation
        """
        self.candidates = list(candidate)
        self.k1 = k1
        self.b = b
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_enabled = cache_enabled
        self.data_source_hash = data_source_hash
        
        # Try to load from cache first
        if self.cache_enabled and self.cache_dir:
            cached_data = self._load_cache()
            if cached_data is not None:
                # Cache loaded successfully
                self._bm25 = cached_data['bm25']
                print(f"✓ Loaded BM25 index from cache ({len(self.candidates)} documents)")
                return
        
        # Build BM25 index from scratch
        print(f"Building BM25 index for {len(self.candidates)} documents...")
        build_start = time.time()
        self._bm25 = BM25Okapi([doc.tokens for doc in self.candidates], k1=k1, b=b)
        build_time = time.time() - build_start
        print(f"✓ BM25 index built in {build_time:.3f}s")
        
        # Save to cache if enabled
        if self.cache_enabled and self.cache_dir:
            self._save_cache()
    
    def _get_cache_path(self) -> Path | None:
        """Get cache file path."""
        if not self.cache_dir:
            return None
        
        # Generate cache filename based on number of documents and parameters
        num_docs = len(self.candidates)
        cache_filename = f"bm25_index_k1_{self.k1}_b_{self.b}_{num_docs}.pkl"
        return self.cache_dir / cache_filename
    
    def _get_cache_metadata_path(self) -> Path | None:
        """Get cache metadata file path."""
        cache_path = self._get_cache_path()
        if cache_path:
            return cache_path.with_suffix('.json')
        return None
    
    def _calculate_data_hash(self) -> str:
        """Calculate hash of candidates data for cache validation."""
        # Create a hash based on candidates IDs and texts
        data_str = "\n".join(f"{c.id}:{c.text}" for c in self.candidates)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()[:16]
    
    def _load_cache(self) -> dict | None:
        """Load BM25 index from cache if available and valid."""
        cache_path = self._get_cache_path()
        metadata_path = self._get_cache_metadata_path()
        
        if not cache_path or not cache_path.exists():
            return None
        
        if not metadata_path or not metadata_path.exists():
            return None
        
        try:
            # Load metadata
            with metadata_path.open('r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate cache
            current_hash = self.data_source_hash or self._calculate_data_hash()
            if metadata.get('data_hash') != current_hash:
                print(f"  ⚠ Cache hash mismatch, will rebuild index")
                return None
            
            if metadata.get('k1') != self.k1 or metadata.get('b') != self.b:
                print(f"  ⚠ BM25 parameters changed, will rebuild index")
                return None
            
            if metadata.get('num_docs') != len(self.candidates):
                print(f"  ⚠ Number of documents changed, will rebuild index")
                return None
            
            # Load BM25 index and candidates
            with cache_path.open('rb') as f:
                cached_data = pickle.load(f)
            
            # Validate candidates match
            cached_candidates = cached_data['candidates']
            if len(cached_candidates) != len(self.candidates):
                print(f"  ⚠ Candidates count mismatch, will rebuild index")
                return None
            
            # Update candidates (in case they have changed, but keep cached tokens)
            # Only update if IDs match
            for i, (cached, current) in enumerate(zip(cached_candidates, self.candidates)):
                if cached.id == current.id:
                    # Use cached tokens but keep current text/name (in case text generation changed)
                    self.candidates[i] = Candidates(
                        id=current.id,
                        name=current.name,
                        text=current.text,
                        tokens=cached.tokens,  # Use cached tokens
                    )
                else:
                    # IDs don't match, cache is invalid
                    print(f"  ⚠ Candidate IDs mismatch, will rebuild index")
                    return None
            
            return {
                'bm25': cached_data['bm25'],
                'candidates': cached_candidates,
            }
            
        except Exception as e:
            print(f"  ⚠ Failed to load cache: {e}")
            print(f"     Will rebuild index...")
            return None
    
    def _save_cache(self) -> None:
        """Save BM25 index and candidates to cache."""
        cache_path = self._get_cache_path()
        metadata_path = self._get_cache_metadata_path()
        
        if not cache_path or not metadata_path:
            return
        
        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate data hash
            data_hash = self.data_source_hash or self._calculate_data_hash()
            
            # Save BM25 index and candidates
            cache_data = {
                'bm25': self._bm25,
                'candidates': self.candidates,
            }
            
            with cache_path.open('wb') as f:
                pickle.dump(cache_data, f)
            
            # Save metadata
            metadata = {
                'data_hash': data_hash,
                'k1': self.k1,
                'b': self.b,
                'num_docs': len(self.candidates),
                'created_at': time.time(),
            }
            
            with metadata_path.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved BM25 cache: {cache_path.name} ({cache_size_mb:.2f} MB)")
            
        except Exception as e:
            print(f"  ⚠ Failed to save cache: {e}")
    # TODO: top_k
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Candidates, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            ((idx, float(score)) for idx, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        results: List[Tuple[Candidates, float]] = []
        for idx, score in ranked:
            if score <= 0:
                break
            results.append((self.candidates[idx], score))
            if len(results) >= top_k:
                break
        return results


def load_food_candidates(
    csv_path: Path,
    cache_dir: Path | str | None = None,
    cache_enabled: bool = True,
) -> tuple[List[Candidates], str]:
    """
    Load food candidates from CSV file.
    
    Args:
        csv_path: Path to CSV file
        cache_dir: Directory to save/load candidates cache (None = use default)
        cache_enabled: Whether to enable caching
        
    Returns:
        Tuple of (candidates list, data_source_hash)
    """
    # Try to load from cache first
    if cache_enabled and cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Generate cache filename based on CSV file
        csv_hash = hashlib.sha256(str(csv_path).encode('utf-8')).hexdigest()[:16]
        cache_filename = f"candidates_{csv_path.stem}_{csv_hash}.pkl"
        cache_path = cache_dir_path / cache_filename
        metadata_path = cache_dir_path / f"{cache_filename}.json"
        
        if cache_path.exists() and metadata_path.exists():
            try:
                # Load metadata
                with metadata_path.open('r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Check if CSV file was modified
                if metadata.get('csv_path') == str(csv_path):
                    csv_mtime = csv_path.stat().st_mtime
                    if metadata.get('csv_mtime') == csv_mtime:
                        # Load cached candidates
                        with cache_path.open('rb') as f:
                            candidates = pickle.load(f)
                        
                        data_hash = metadata.get('data_hash', '')
                        print(f"✓ Loaded {len(candidates)} candidates from cache")
                        return candidates, data_hash
            except Exception as e:
                print(f"  ⚠ Failed to load candidates cache: {e}")
                print(f"     Will rebuild from CSV...")
    
    # Load from CSV
    print(f"Loading candidates from CSV: {csv_path}")
    load_start = time.time()
    candidates: List[Candidates] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("itemMetadata")
            if not raw:
                continue
            metadata = json.loads(raw)
            text = build_candidate_text(metadata)
            tokens = tokenize(text)
            if not tokens:
                continue
            candidates.append(
                Candidates(
                    id=row.get("itemId", ""),
                    name=metadata.get("name", ""),
                    text=text,
                    tokens=tokens,
                )
            )
    
    load_time = time.time() - load_start
    print(f"✓ Loaded {len(candidates)} candidates from CSV in {load_time:.3f}s")
    
    # Calculate data hash
    data_hash = hashlib.sha256(
        "\n".join(f"{c.id}:{c.text}" for c in candidates).encode('utf-8')
    ).hexdigest()[:16]
    
    # Save to cache if enabled
    if cache_enabled and cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        
        csv_hash = hashlib.sha256(str(csv_path).encode('utf-8')).hexdigest()[:16]
        cache_filename = f"candidates_{csv_path.stem}_{csv_hash}.pkl"
        cache_path = cache_dir_path / cache_filename
        metadata_path = cache_dir_path / f"{cache_filename}.json"
        
        try:
            # Save candidates
            with cache_path.open('wb') as f:
                pickle.dump(candidates, f)
            
            # Save metadata
            metadata = {
                'csv_path': str(csv_path),
                'csv_mtime': csv_path.stat().st_mtime,
                'data_hash': data_hash,
                'num_candidates': len(candidates),
                'created_at': time.time(),
            }
            
            with metadata_path.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved candidates cache: {cache_path.name} ({cache_size_mb:.2f} MB)")
        except Exception as e:
            print(f"  ⚠ Failed to save candidates cache: {e}")
    
    return candidates, data_hash


def demonstrate_sample_query(data_dir: Path) -> None:
    # TODO: env parameter config
    items_path = data_dir / "5k_items_curated.csv"
    queries_path = data_dir / "queries.csv"

    candidates, _ = load_food_candidates(items_path)
    retriever = BM25Retriever(candidates)

    with queries_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader) # only retreive the first row

    query = first_row["search_term_pt"]
    results = retriever.search(query, top_k=5)

    print(f"sample: {query}")
    print("\nTop-5 results:")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}. score={score:.4f} itemId={doc.id} name={doc.name} tokens={doc.tokens}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    demonstrate_sample_query(project_root / "data" / "raw")

