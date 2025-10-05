# scripts/peek_nodes.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db import SessionLocal
from sqlalchemy import text

SRC_CANDS = ["cause","subject","src","source","entity1","node1","from_node"]
DST_CANDS = ["effect","result","object","dst","target","entity2","node2","to_node","outcome"]

def pragma_columns(session, table="triple"):
    rows = session.execute(text(f"PRAGMA table_info({table})")).fetchall()
    # rows: (cid, name, type, notnull, dflt_value, pk)
    return [(r[1], (r[2] or "").upper()) for r in rows]

def existing(cols, cands):
    # keep order from cands; return subset that exist in table
    lower = {c[0].lower(): c[0] for c in cols}
    out = []
    for c in cands:
        if c.lower() in lower:
            out.append(lower[c.lower()])
    return out

def text_like(cols):
    # fallback: return any TEXT-ish columns except id/evidence/label/relation
    names = []
    for name, typ in cols:
        nl = name.lower()
        if nl in {"id"}: continue
        if nl.startswith("evidence_"): continue
        if nl in {"relation","rel","edge","edge_type","type"}: continue
        if "CHAR" in typ or "TEXT" in typ or typ == "":
            names.append(name)
    return names

def main():
    s = SessionLocal()

    cols = pragma_columns(s, "triple")
    if not cols:
        print("[err] table 'triple' not found"); return

    src_cols = existing(cols, SRC_CANDS)
    dst_cols = existing(cols, DST_CANDS)
    # Fallback if nothing matched
    if not src_cols and not dst_cols:
        src_cols = text_like(cols)

    selects = []
    for c in (src_cols + dst_cols):
        selects.append(f"SELECT {c} AS value FROM triple")

    if not selects:
        print("[err] no usable columns to sample node names"); return

    union_sql = " UNION ".join(selects)
    sql = f"""
        SELECT DISTINCT value FROM (
            {union_sql}
        )
        WHERE value IS NOT NULL AND value <> ''
        LIMIT 20
    """

    rows = s.execute(text(sql)).fetchall()
    for (v,) in rows:
        print(v)
    s.close()

if __name__ == "__main__":
    main()
