# --- Standard library ---
from collections import Counter, defaultdict

# --- Third-party ---
import spacy

# --- Local modules ---
from db import SessionLocal
from models import Publication, Entity, Triple

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------------------------
# Filtering + Normalization
# -------------------------------------------------------------------


def normalize_entity(text: str) -> str:
    """Return a clean, noun-phrase style version of the entity."""
    if not text:
        return ""
    doc = nlp(text.strip())
    # Prefer noun chunks if they exist
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    if noun_chunks:
        return noun_chunks[0].lower().strip()
    # Otherwise, fall back to noun/proper noun tokens
    tokens = [t.text for t in doc if t.pos_ in {
        "NOUN", "PROPN"} and not t.is_stop]
    return " ".join(tokens).lower().strip() if tokens else text.lower().strip()


def normalize_relation(text: str) -> str:
    """Return a clean relation phrase (verb + object if possible)."""
    if not text:
        return ""
    doc = nlp(text.strip())
    verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"]
    objs = [t.lemma_ for t in doc if t.dep_ in {"dobj", "pobj", "attr"}]
    if verbs and objs:
        return f"{verbs[0]} {objs[0]}".lower().strip()
    if verbs:
        return verbs[0].lower().strip()
    return text.lower().strip()


def is_valid_entity(text: str, etype: str | None) -> bool:
    if not text:
        return False
    doc = nlp(text.strip())
    if not any(t.pos_ in {"NOUN", "PROPN"} for t in doc):
        return False
    if all(t.is_stop for t in doc):
        return False
    if etype and etype.lower() not in {
        "organism", "gene", "protein", "tissue", "condition", "outcome"
    }:
        return False
    return True


def is_valid_relation(text: str) -> bool:
    if not text:
        return False
    doc = nlp(text.strip())
    verbs = [t for t in doc if t.pos_ == "VERB"]
    nouns = [t for t in doc if t.pos_ in {"NOUN", "PROPN"}]
    if not verbs or not nouns:
        return False
    if len(doc) == 1 and doc[0].pos_ == "VERB":
        return False
    return True

# -------------------------------------------------------------------
# Trends
# -------------------------------------------------------------------


def compute_entity_trends() -> dict:
    """Count normalized entities per year."""
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
                norm = normalize_entity(text)
                if norm:
                    trends_by_year[year][norm] += 1
    finally:
        db.close()
    return {year: dict(counter) for year, counter in trends_by_year.items()}


def compute_relation_trends() -> dict:
    """Count normalized relations per year."""
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
                norm = normalize_relation(relation)
                if norm:
                    trends_by_year[year][norm] += 1
    finally:
        db.close()
    return {year: dict(counter) for year, counter in trends_by_year.items()}

# -------------------------------------------------------------------
# Top-N Wrapper
# -------------------------------------------------------------------


def compute_top_trends(trends_by_year: dict, top_n: int = 10, label: str = "items"):
    """Return top-N per year with a configurable label."""
    results = []
    for year, counter in sorted(trends_by_year.items()):
        top_items = [
            {"name": term, "frequency": count}
            for term, count in Counter(counter).most_common(top_n)
        ]
        results.append({"year": year, label: top_items})
    return results
