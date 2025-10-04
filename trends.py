from collections import Counter, defaultdict
from db import SessionLocal
from models import Publication
from sqlalchemy import func

from models import Entity, Triple


STOP_ENTITIES = {"is", "are", "was", "were", "have",
                 "has", "like", "increases", "decreases", "include"}


def compute_entity_trends() -> dict:
    """Compute entity frequency per year directly from DB (with cleanup)."""
    db = SessionLocal()
    trends_by_year = defaultdict(Counter)
    try:
        rows = (
            db.query(Publication.year, Entity.text)
            .join(Entity, Entity.publication_id == Publication.id)
            .filter(Publication.year.isnot(None))
            .all()
        )
        for year, text in rows:
            if not text:
                continue
            term = text.strip().lower()
            if term in STOP_ENTITIES or len(term) < 3:
                continue
            trends_by_year[year][term] += 1
    finally:
        db.close()
    return {year: dict(counter) for year, counter in trends_by_year.items()}


def compute_top_trends(trends_by_year: dict, top_n: int = 10):
    """Return top entities per year."""
    results = []
    for year, counter in sorted(trends_by_year.items()):
        top_entities = [
            {"term": term, "count": count}
            for term, count in Counter(counter).most_common(top_n)
        ]
        results.append({"year": year, "top_entities": top_entities})
    return results


def compute_relation_trends() -> dict:
    """Compute relationship (triple) frequency per year."""
    db = SessionLocal()
    trends_by_year = defaultdict(Counter)
    try:
        rows = (
            db.query(Publication.year, Triple.relation)
            .join(Triple, Triple.publication_id == Publication.id)
            .filter(Publication.year.isnot(None))
            .all()
        )
        for year, relation in rows:
            if relation:
                trends_by_year[year][relation.lower()] += 1
    finally:
        db.close()
    return {year: dict(counter) for year, counter in trends_by_year.items()}


def compute_gaps(min_threshold: int = 5, relative: float = 0.2):
    """
    Detect research gaps by finding entities that are underrepresented.
    - min_threshold: absolute minimum number of mentions to be considered well-studied
    - relative: fraction of median count to flag as a gap
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(Entity.text)
            .join(Publication, Entity.publication_id == Publication.id)
            .filter(Publication.year.isnot(None))
            .all()
        )

        counts = Counter([r[0].lower() for r in rows if r[0]])
        if not counts:
            return []

        # median for relative comparison
        values = list(counts.values())
        median_val = sorted(values)[len(values)//2]

        gaps = []
        for term, count in counts.items():
            if count < min_threshold or count < median_val * relative:
                gaps.append({"term": term, "count": count})

        # sort by ascending frequency (rarest first)
        gaps = sorted(gaps, key=lambda x: x["count"])
        return gaps
    finally:
        db.close()
