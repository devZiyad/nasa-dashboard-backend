# ml/retriever_infer.py
"""
Lightweight retriever:
- Loads a local SentenceTransformer model from artifacts/retriever
- Builds an in-memory embedding index for doc passages
- Runs cosine-sim search and returns (id, score) pairs
"""

import os
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# --- Globals (in-memory cache) ---
_model: SentenceTransformer | None = None
_index_embeddings: np.ndarray | None = None
_index_ids: List[int] | None = None


def load_model(path: str = "artifacts/retriever") -> SentenceTransformer:
    """
    Load the SentenceTransformer from a LOCAL folder.
    We resolve to an absolute path so SentenceTransformer never treats it
    as a Hugging Face model ID (which causes 401/NotFound errors).
    """
    global _model
    if _model is None:
        abs_path = os.path.abspath(path)
        print(f"[retriever] loading LOCAL model from: {abs_path}")
        _model = SentenceTransformer(abs_path)
    return _model


def build_index(texts: List[str], ids: List[int]) -> None:
    """
    Build the in-memory index.
    texts: doc passages parallel to ids
    ids:   integer ids (e.g., publication ids)
    """
    model = load_model()
    # Normalize embeddings so we can use dot-product == cosine similarity
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=128
    )

    global _index_embeddings, _index_ids
    _index_embeddings = embs
    _index_ids = list(ids)


def search(query: str, k: int = 10) -> List[Tuple[int, float]]:
    """
    Encode the query, compute cosine sims against the in-memory index,
    and return top-k (id, score) pairs sorted by score desc.
    """
    if not query:
        return []
    if _index_embeddings is None or _index_ids is None or len(_index_embeddings) == 0:
        return []

    model = load_model()
    q = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]  # shape (d,)

    # cosine sims since both sides are normalized
    sims = _index_embeddings @ q  # (N,)

    k = max(1, min(k, len(sims)))
    # partial top-k for speed
    topk_idx = np.argpartition(-sims, k - 1)[:k]
    # sort those k by actual score
    order = topk_idx[np.argsort(-sims[topk_idx])]

    return [(int(_index_ids[i]), float(sims[i])) for i in order]
