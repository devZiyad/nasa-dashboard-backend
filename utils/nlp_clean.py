# utils/nlp_clean.py
import spacy

# Confidence mapping
CONFIDENCE_MAP = {
    "high": 0.95,
    "medium": 0.6,
    "low": 0.3
}

# Load spaCy model (try SciSpacy first for biomedical text)
try:
    nlp = spacy.load("en_core_sci_sm")
except:
    nlp = spacy.load("en_core_web_sm")


# ------------------
# Entity helpers
# ------------------
def normalize_entity(text: str) -> str:
    """Return cleaned entity using noun chunks or lemmatized nouns."""
    if not text:
        return ""
    doc = nlp(text)
    chunks = [chunk.text.lower().strip()
              for chunk in doc.noun_chunks if len(chunk.text) > 2]
    if chunks:
        return min(chunks, key=len)  # pick shortest noun chunk
    nouns = [t.lemma_.lower() for t in doc if t.pos_ in {"NOUN", "PROPN"}]
    return " ".join(nouns) if nouns else text.lower().strip()


def is_valid_entity(text: str, etype: str | None = None) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 3:
        return False
    doc = nlp(t)
    if not any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc):
        return False
    if all(tok.is_stop for tok in doc):
        return False
    if etype and etype.lower() not in {
        "organism", "gene", "protein", "tissue", "condition", "outcome"
    }:
        return False
    return True


# ------------------
# Relation helpers
# ------------------
def normalize_relation(text: str) -> str:
    """Return simplified relation (main verb lemma)."""
    if not text:
        return ""
    doc = nlp(text)
    verbs = [t.lemma_.lower() for t in doc if t.pos_ == "VERB"]
    return verbs[0] if verbs else text.lower().strip()


def is_valid_relation(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 3:
        return False
    doc = nlp(t)
    verbs = [tok for tok in doc if tok.pos_ == "VERB"]
    nouns = [tok for tok in doc if tok.pos_ in {"NOUN", "PROPN"}]
    if not verbs or not nouns:
        return False
    if len(doc) == 1 and doc[0].pos_ == "VERB":  # reject bare verbs
        return False
    return True


# ------------------
# Confidence normalization
# ------------------
def normalize_confidence(val):
    """Normalize confidence scores from text/float/int to float [0-1]."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_lower = val.strip().lower()
        if val_lower in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[val_lower]
        try:
            return float(val)
        except ValueError:
            return None
    return None
