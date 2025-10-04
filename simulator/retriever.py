from typing import List, Dict, Optional
from .indexer import assemble_docs, Index
from .config import RESULTS_BOOST, DISCUSSION_BOOST
from .db import connect

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

def fallback_sql_like(query: str, k: int = 5):
    """
    Lightweight LIKE-based fallback if TF-IDF returns nothing.
    Searches 'section.text' + returns minimal doc metadata by joining publication.
    """
    terms = [t.strip() for t in query.split() if t.strip()]
    if not terms:
        return []
    like_clauses = " AND ".join(["s.text LIKE ?"] * len(terms))
    args = [f"%{t}%" for t in terms]

    sql = f"""
    SELECT p.id AS publication_id, p.title, p.year, p.journal, p.link, s.kind, s.text
    FROM section s
    JOIN publication p ON p.id = s.publication_id
    WHERE {like_clauses}
    LIMIT ?
    """
    args.append(k * 2)

    rows = []
    with connect() as con:
        for r in con.execute(sql, args):
            snippet = (r["text"] or "")[:500]
            rows.append({
                "publication_id": r["publication_id"],
                "title": r["title"],
                "year": r["year"],
                "journal": r["journal"],
                "link": r["link"],
                "snippet": f"[{(r['kind'] or '').upper()}]\n{snippet}",
                "score": 0.08,  # conservative low score (below 'balanced' threshold)
                "entities": {}, # we keep it empty here
            })
    return rows[:k]