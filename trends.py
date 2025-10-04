from collections import Counter, defaultdict
from db import SessionLocal
from models import Publication, Entity, Triple
import spacy

# Load spaCy NLP model (small English model is fine for POS/deps)
nlp = spacy.load("en_core_web_sm")

# --- Entity Filtering ---


def is_valid_entity(text: str, etype: str | None) -> bool:
    """Check if an entity is valid using spaCy POS rules and optional type restrictions."""
    if not text:
        return False
    text = text.strip()
    if len(text) < 3:
        return False

    doc = nlp(text)

    # At least one noun or proper noun required
    if not any(t.pos_ in {"NOUN", "PROPN"} for t in doc):
        return False

    # Filter out purely stopword phrases (like "of", "with")
    if all(t.is_stop for t in doc):
        return False

    # Optional: enforce entity types if extractor provides them
    if etype and etype.lower() not in {
        "organism", "gene", "protein", "tissue", "condition", "outcome"
    }:
        return False

    return True

# --- Relation Filtering ---


def is_valid_relation(text: str) -> bool:
    """Check if a relation is valid: must have a verb and a noun (so 'induces apoptosis' is ok, 'induces' is not)."""
    if not text:
        return False
    text = text.strip()
    if len(text) < 3:
        return False

    doc = nlp(text)

    verbs = [t for t in doc if t.pos_ == "VERB"]
    nouns = [t for t in doc if t.pos_ in {"NOUN", "PROPN"}]

    # Require at least one verb and one noun
    if not verbs or not nouns:
        return False

    # Drop trivial single-token verbs (like "induces")
    if len(doc) == 1 and doc[0].pos_ == "VERB":
        return False

    return True

# --- Entity Trends ---


def compute_entity_trends() -> dict:
    db = SessionLocal()
    trends_by_year = defaultdict(Counter)
    try:
        rows = (
            db.query(Publication.year, Entity.text, Entity.type)
            .join(Entity, Entity.publication_id == Publication.id)
            .filter(Publication.year.isnot(None))
            .all()
        )
        for year, text, etype in rows:
            if is_valid_entity(text, etype):
                trends_by_year[year][text.strip().lower()] += 1
    finally:
        db.close()
    return {year: dict(counter) for year, counter in trends_by_year.items()}

# --- Relation Trends ---


def compute_relation_trends() -> dict:
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
            if relation and is_valid_relation(relation):
                trends_by_year[year][relation.strip().lower()] += 1
    finally:
        db.close()
    return {year: dict(counter) for year, counter in trends_by_year.items()}

# --- Top Entities/Relations ---


def compute_top_trends(trends_by_year: dict, top_n: int = 10):
    results = []
    for year, counter in sorted(trends_by_year.items()):
        top_items = [
            {"term": term, "count": count}
            for term, count in Counter(counter).most_common(top_n)
        ]
        results.append({"year": year, "top_entities": top_items})
    return results
