from typing import List, Dict, Optional
from .indexer import assemble_docs, Index
from .config import RESULTS_BOOST, DISCUSSION_BOOST

INDEX: Optional[Index] = None

def ensure_index() -> Index:
    global INDEX
    if INDEX is None:
        docs = assemble_docs()
        INDEX = Index(docs)
    return INDEX

def boost_score(snippet: str, score: float) -> float:
    if "[RESULTS]" in snippet:
        score *= RESULTS_BOOST
    if "[DISCUSSION]" in snippet:
        score *= DISCUSSION_BOOST
    return score

def search(query: str, k: int = 10, filters: Dict = None) -> List[Dict]:
    idx = ensure_index()
    raw = idx.search(query, k=k*3)
    filters = filters or {}
    etypes = filters.get("entity_types") or {}
    year_min = filters.get("year_min")

    def ok(r: Dict) -> bool:
        if year_min and isinstance(r.get("year"), int) and r["year"] < year_min:
            return False
        for etype, required in etypes.items():
            if required and set(map(str.lower, r.get("entities", {}).get(etype, []))).isdisjoint({x.lower() for x in required}):
                return False
        return True

    kept = []
    for r in raw:
        if ok(r):
            r["score"] = boost_score(r["snippet"], r["score"])
            kept.append(r)
    kept.sort(key=lambda x: x["score"], reverse=True)
    return kept[:k]
