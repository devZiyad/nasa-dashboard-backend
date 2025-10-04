from collections import Counter, defaultdict
from db import SessionLocal
from models import Publication
from sqlalchemy import func

from models import Entity, Triple  

def compute_entity_trends() -> dict:
    """Compute entity frequency per year directly from DB."""
    db = SessionLocal()
    trends_by_year = defaultdict(Counter)
    try:
        # join entities with publications
        rows = (
            db.query(Publication.year, Entity.text)
            .join(Entity, Entity.publication_id == Publication.id)
            .filter(Publication.year.isnot(None))
            .all()
        )

        for year, text in rows:
            if text:
                trends_by_year[year][text.lower()] += 1

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
