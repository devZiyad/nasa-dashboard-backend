from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
_index_embeddings = None
_index_ids = None

def load_model(path="artifacts/retriever"):
    global _model
    if _model is None:
        _model = SentenceTransformer(path)
    return _model

def build_index(texts, ids):
    # texts: list[str] (doc passages); ids parallel to texts
    model = load_model()
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
    global _index_embeddings, _index_ids
    _index_embeddings = embs
    _index_ids = ids

def search(query, k=10):
    model = load_model()
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = (_index_embeddings @ q)
    topk = np.argpartition(-sims, k-1)[:k]
    order = topk[np.argsort(-sims[topk])]
    return [(int(_index_ids[i]), float(sims[i])) for i in order]
