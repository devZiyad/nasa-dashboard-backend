from typing import Dict, List
from .db import connect, TRIPLES_SQL

def graph_query(q: str, limit: int = 200) -> Dict:
    terms = [t.strip() for t in q.split() if t.strip()]
    where = []
    args = []
    for t in terms:
        where.append("(subject LIKE ? OR object LIKE ?)")
        like = f"%{t}%"
        args.extend([like, like])
    sql = TRIPLES_SQL + (" WHERE " + " AND ".join(where) if where else "") + " LIMIT ?"
    args.append(limit)

    nodes = {}
    edges = []
    with connect() as con:
        for row in con.execute(sql, args):
            s, o, r = row["subject"], row["object"], row["relation"]
            nodes.setdefault(s, {"id": s, "type": "term"})
            nodes.setdefault(o, {"id": o, "type": "term"})
            edges.append({
                "source": s,
                "target": o,
                "relation": r,
                "weight": float(row["confidence"] or 0.5),
                "evidence": [{
                    "publication_id": row["publication_id"],
                    "evidence_sentence": row["evidence_sentence"],
                    "confidence": float(row["confidence"] or 0.5),
                }],
            })
    return {"nodes": list(nodes.values()), "edges": edges}
