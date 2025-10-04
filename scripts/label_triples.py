# scripts/label_triples.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root

from db import SessionLocal
from sqlalchemy import text
from ml.evidence_classifier import predict

# Column aliases we’ll try in your `triple` table
CAND_CAUSE    = ["cause", "subject", "src", "source", "entity1", "node1", "from_node"]
CAND_RELATION = ["relation", "predicate", "rel", "edge", "type", "edge_type"]
CAND_EFFECT   = ["effect", "object", "dst", "target", "entity2", "node2", "to_node"]
CAND_CONTEXT  = ["context", "evidence", "sentence", "text", "snippet", "note", "detail"]

def pragma_columns(session, table="triple"):
    rows = session.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return [r[1] for r in rows]  # [name, ...]

def pick(cols, candidates):
    cset = [c.lower() for c in candidates]
    # exact match first
    for c in cols:
        if c.lower() in cset:
            return c
    # fuzzy contains next
    for c in cols:
        cl = c.lower()
        if any(x in cl for x in cset):
            return c
    return None

def ensure_columns(session):
    cols = set(pragma_columns(session, "triple"))
    stmts = []
    if "evidence_label" not in cols:
        stmts.append("ALTER TABLE triple ADD COLUMN evidence_label TEXT")
    if "evidence_prob" not in cols:
        stmts.append("ALTER TABLE triple ADD COLUMN evidence_prob REAL")
    for s in stmts:
        session.execute(text(s))
    if stmts:
        session.commit()

def main():
    session = SessionLocal()
    ensure_columns(session)

    cols = pragma_columns(session, "triple")
    if not cols:
        raise RuntimeError("Table 'triple' not found.")

    id_col = "id" if "id" in cols else cols[0]

    cause_col    = pick(cols, CAND_CAUSE)
    relation_col = pick(cols, CAND_RELATION)
    effect_col   = pick(cols, CAND_EFFECT)
    context_col  = pick(cols, CAND_CONTEXT)

    select_cols = [id_col]
    for c in [cause_col, relation_col, effect_col, context_col]:
        if c and c not in select_cols:
            select_cols.append(c)

    sel_sql = "SELECT " + ", ".join(select_cols) + " FROM triple"
    rows = session.execute(text(sel_sql)).fetchall()
    if not rows:
        print("[warn] No rows in 'triple'.")
        session.close()
        return

    # Build texts
    texts, ids = [], []
    for r in rows:
        row = dict(zip(select_cols, r))
        ids.append(row[id_col])

        ctx = (row.get(context_col) or "").strip() if context_col else ""
        if ctx:
            texts.append(ctx)
            continue

        parts = []
        if cause_col:    parts.append(str(row.get(cause_col, "") or ""))
        if relation_col: parts.append(str(row.get(relation_col, "") or ""))
        if effect_col:   parts.append(str(row.get(effect_col, "") or ""))
        txt = " ".join([p for p in parts if p]).strip()
        if txt and not txt.endswith("."):
            txt += "."
        texts.append(txt or "evidence.")

    print(f"[info] predicting evidence strength for {len(texts)} rows...")
    probs = predict(texts)  # P(strong)

    # Write back
    updated = 0
    for rid, p in zip(ids, probs):
        label = "strong" if p >= 0.60 else "weak"
        session.execute(
            text("UPDATE triple SET evidence_label=:label, evidence_prob=:p WHERE {}=:id".format(id_col)),
            {"label": label, "p": float(p), "id": rid}
        )
        updated += 1

    session.commit()
    session.close()
    print(f"[ok] updated {updated} rows in 'triple' with evidence_label + evidence_prob")

if __name__ == "__main__":
    main()
