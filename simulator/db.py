import sqlite3
from contextlib import contextmanager
from .config import DB_PATH

@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()

DOC_KINDS_SQL = """
SELECT p.id AS publication_id, p.title, p.year, p.journal, p.link,
       s.kind, s.text
FROM publication p
JOIN section s ON s.publication_id = p.id
WHERE s.kind IN ({placeholders})
ORDER BY p.id
"""

ENTITIES_BY_PUB_SQL = """
SELECT publication_id, type, text FROM entity
"""

TRIPLES_SQL = """
SELECT publication_id, subject, relation, object, evidence_sentence, confidence
FROM triple
"""

# --- DB-driven schema helpers ---

def distinct_entities_by_type(entity_type: str, limit: int = 200):
    sql = """
    SELECT DISTINCT value
    FROM entity
    WHERE LOWER(type) = LOWER(?)
    ORDER BY value
    LIMIT ?
    """
    with connect() as con:
        return [r["value"] for r in con.execute(sql, (entity_type, limit))]

def distinct_interventions(limit: int = 200):
    # support either 'intervention' or 'countermeasure' labels
    sql = """
    SELECT DISTINCT value
    FROM entity
    WHERE LOWER(type) IN ('intervention', 'countermeasure')
    ORDER BY value
    LIMIT ?
    """
    with connect() as con:
        return [r["value"] for r in con.execute(sql, (limit,))]

def outcome_alias_rows(limit: int = 1000):
    """
    Optional: if you later create a small outcome_alias table:
      CREATE TABLE IF NOT EXISTS outcome_alias (
        outcome TEXT, alias TEXT
      );
    We’ll read from it here. If it doesn't exist, return [].
    """
    try:
        sql = "SELECT outcome, alias FROM outcome_alias LIMIT ?"
        with connect() as con:
            return [{"outcome": r["outcome"], "alias": r["alias"]}
                    for r in con.execute(sql, (limit,))]
    except Exception:
        return []
