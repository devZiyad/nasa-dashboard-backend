from typing import Dict, List
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .db import connect, DOC_KINDS_SQL, ENTITIES_BY_PUB_SQL
from .config import SECTION_POLICY, MIN_CHARS

class Index:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.texts = [d["text"] for d in docs]
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 10) -> List[Dict]:
        if not query.strip():
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            d = self.docs[int(i)]
            out.append({
                "score": float(sims[int(i)]),
                **{k:v for k,v in d.items() if k != "text"},
                "snippet": d["text"][:500],
            })
        return out

def assemble_docs() -> List[Dict]:
    kinds = ",".join(["?" for _ in SECTION_POLICY])
    sql = DOC_KINDS_SQL.format(placeholders=kinds)

    by_pub = {}
    blob = defaultdict(list)
    with connect() as con:
        for row in con.execute(sql, SECTION_POLICY):
            pid = row["publication_id"]
            if pid not in by_pub:
                by_pub[pid] = {
                    "publication_id": pid,
                    "title": row["title"],
                    "year": row["year"],
                    "journal": row["journal"],
                    "link": row["link"],
                }
            text = (row["text"] or "").strip()
            if text:
                blob[pid].append(f"[{row['kind'].upper()}]\n{text}")

        facets = defaultdict(lambda: defaultdict(set))
        for er in con.execute(ENTITIES_BY_PUB_SQL):
            facets[er["publication_id"]][er["type"]].add(er["text"])

    docs = []
    for pid, meta in by_pub.items():
        combined = "\n\n".join(blob[pid]).strip()
        if len(combined) < MIN_CHARS:
            continue
        docs.append({
            "text": combined,
            **meta,
            "entities": {t: sorted(list(vals)) for t, vals in facets[pid].items()}
        })
    return docs
