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

