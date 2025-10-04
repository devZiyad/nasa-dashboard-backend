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

    # Prefer noun chunks
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    if noun_chunks:
        return noun_chunks[0].lower().strip()

    # Fallback: nouns/proper nouns
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
    """Check that the entity is linguistically valid."""
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
    """Check that the relation is linguistically valid."""
    if not text:
        return False
    doc = nlp(text.strip())

    verbs = [t for t in doc if t.pos_ == "VERB"]
    nouns = [t for t in doc if t.pos_ in {"NOUN", "PROPN"}]

    if not verbs or not nouns:
        return False
    if len(doc) == 1 and doc[0].pos_ == "VERB":  # reject bare verbs
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
            db.query(Publication.year, Entity.normalized_text,
                     Entity.text, Entity.type)
            .join(Entity, Entity.publication_id == Publication.id)
            .filter(Publication.year.isnot(None))
            .all()
        )

        for year, norm_text, raw_text, etype in rows:
            candidate = norm_text or raw_text
            if candidate and is_valid_entity(candidate, etype):
                norm = normalize_entity(candidate)
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
            db.query(Publication.year,
                     Triple.normalized_relation, Triple.relation)
            .join(Triple, Triple.publication_id == Publication.id)
            .filter(Publication.year.isnot(None))
            .all()
        )

        for year, norm_rel, raw_rel in rows:
            candidate = norm_rel or raw_rel
            if candidate and is_valid_relation(candidate):
                norm = normalize_relation(candidate)
                if norm:
                    trends_by_year[year][norm] += 1
    finally:
        db.close()
    return {year: dict(counter) for year, counter in trends_by_year.items()}


def compute_gaps(min_threshold: int = 5, relative: float = 0.2):
    """
    Detect underrepresented entities compared to the average frequency.
    - min_threshold: skip entities that appear fewer than this.
    - relative: mark entities as "gaps" if below relative * avg frequency.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(Entity.normalized_text, Entity.text)
            .join(Publication, Entity.publication_id == Publication.id)
            .filter(Publication.year.isnot(None))
            .all()
        )

        # Prefer normalized text, else raw
        texts = [r[0] or r[1] for r in rows if (r[0] or r[1])]
        counts = Counter([t.strip().lower() for t in texts if t])
        if not counts:
            return []

        avg_freq = sum(counts.values()) / len(counts)

        gaps = [
            {"name": term, "frequency": freq}
            for term, freq in counts.items()
            if freq < min_threshold or freq < relative * avg_freq
        ]

        gaps.sort(key=lambda x: x["frequency"])
        return gaps
    finally:
        db.close()

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
