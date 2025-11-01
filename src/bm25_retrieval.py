from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(
    r"[0-9]+(?:\.[0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ0-9_']+"
)

# do not sepereate .numbers, not_vegan, words
def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_document_text(metadata: dict) -> str:
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
        pieces.append("lactose_free")
    elif metadata.get("lacFree") is False:
        pieces.append("not_lactose_free")

    if metadata.get("organic") is True:
        pieces.append("organic")
    elif metadata.get("organic") is False:
        pieces.append("not_organic")

    if metadata.get("vegan") is True:
        pieces.append("vegan")
    elif metadata.get("vegan") is False:
        pieces.append("not_vegan")

    return " ".join(pieces)


@dataclass
class Document:
    doc_id: str
    name: str
    text: str
    tokens: List[str]


class BM25Retriever:
    # TODO: k1, b
    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self._bm25 = BM25Okapi([doc.tokens for doc in self.documents], k1=k1, b=b)
    # TODO: top_k
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            ((idx, float(score)) for idx, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        results: List[Tuple[Document, float]] = []
        for idx, score in ranked:
            if score <= 0:
                break
            results.append((self.documents[idx], score))
            if len(results) >= top_k:
                break
        return results


def load_food_documents(csv_path: Path) -> List[Document]:
    documents: List[Document] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("itemMetadata")
            if not raw:
                continue
            metadata = json.loads(raw)
            text = build_document_text(metadata)
            tokens = tokenize(text)
            if not tokens:
                continue
            documents.append(
                Document(
                    doc_id=row.get("itemId", ""),
                    name=metadata.get("name", ""),
                    text=text,
                    tokens=tokens,
                )
            )
    return documents


def demonstrate_sample_query(data_dir: Path) -> None:
    items_path = data_dir / "5k_items_curated.csv"
    queries_path = data_dir / "queries.csv"

    documents = load_food_documents(items_path)
    retriever = BM25Retriever(documents)

    with queries_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader) # only retreive the first row

    query = first_row["search_term_pt"]
    results = retriever.search(query, top_k=5)

    print(f"sample: {query}")
    print("\nTop-5 results:")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}. score={score:.4f} itemId={doc.doc_id} name={doc.name} tokens={doc.tokens}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    demonstrate_sample_query(project_root / "data" / "raw")

