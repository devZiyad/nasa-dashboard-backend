# scripts/prepare_evidence_seed.py
import re, csv, sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root

from db import SessionLocal
from sqlalchemy import text

# --- Heuristics for column aliases ---
CAND_CAUSE    = ["cause", "subject", "src", "source", "entity1", "node1", "from_node"]
CAND_RELATION = ["relation", "predicate", "rel", "edge", "type", "edge_type"]
CAND_EFFECT   = ["effect", "object", "dst", "target", "entity2", "node2", "to_node"]
CAND_CONTEXT  = ["context", "evidence", "sentence", "text", "abstract", "summary", "snippet"]

STRONG_PATTERNS = [
    r"\bp\s*<\s*0\.05\b", r"\bp\s*<\s*0\.01\b", r"\bsignificant(ly)?\b",
    r"\brobust\b", r"\breplicated\b", r"\bconsistent\b", r"\bmeta[- ]?analysis\b",
    r"\brandomized\b", r"\bcontrolled\b", r"\bclinical\b", r"\bhuman (study|trial)\b"
]
WEAK_PATTERNS = [
    r"\bin vitro\b", r"\bin silico\b", r"\bpilot\b", r"\bprelim(inary)?\b",
    r"\bsmall sample\b", r"\banimal (model|study)\b", r"\bmouse\b", r"\bmurine\b",
    r"\brat\b", r"\bcorrelat(es?|ion)\b", r"\btrend\b", r"\bmay\b", r"\bmight\b", r"\bcould\b"
]

def label_from_text(text: str) -> int:
    t = (text or "").lower()
    strong = any(re.search(p, t) for p in STRONG_PATTERNS)
    weak   = any(re.search(p, t) for p in WEAK_PATTERNS)
    if strong and not weak: return 1
    if weak and not strong: return 0
    if strong and weak: return 1
    return 0

def pragma_columns(session, table="triple"):
    rows = session.execute(text(f"PRAGMA table_info({table})")).fetchall()
    # rows: cid, name, type, notnull, dflt_value, pk
    return [r[1] for r in rows]

def pick(cols, candidates):
    cand_lc = [c.lower() for c in candidates]
    for c in cols:
        cl = c.lower()
        if cl in cand_lc:
            return c
    # fuzzy contains
    for c in cols:
        cl = c.lower()
        if any(cl == x or x in cl for x in cand_lc):
            return c
    return None

def main():
    session = SessionLocal()

    # --- detect actual column names ---
    cols = pragma_columns(session, "triple")
    if not cols:
        raise RuntimeError("Table 'triple' not found.")

    id_col = "id" if "id" in cols else cols[0]  # fall back to first col as an id
    cause_col    = pick(cols, CAND_CAUSE)
    relation_col = pick(cols, CAND_RELATION)
    effect_col   = pick(cols, CAND_EFFECT)
    context_col  = pick(cols, CAND_CONTEXT)

    # Build SELECT list dynamically
    select_cols = [id_col]
    if cause_col:    select_cols.append(cause_col)
    if relation_col: select_cols.append(relation_col)
    if effect_col:   select_cols.append(effect_col)
    if context_col:  select_cols.append(context_col)

    sel_sql = "SELECT " + ", ".join(select_cols) + " FROM triple"
    rows = session.execute(text(sel_sql)).fetchall()

    out_rows = []
    for r in rows:
        row = dict(zip(select_cols, r))

        cause    = str(row.get(cause_col, "") or "")
        relation = str(row.get(relation_col, "") or "")
        effect   = str(row.get(effect_col, "") or "")
        ctx      = str(row.get(context_col, "") or "")

        if ctx.strip():
            text_val = ctx.strip()
        else:
            # synthesize a compact sentence from whatever we have
            parts = [p for p in [cause, relation, effect] if p]
            text_val = " ".join(parts).strip()
            if text_val and not text_val.endswith("."):
                text_val += "."

        if not text_val:
            # skip empty rows gracefully
            continue

        label = label_from_text(text_val)
        out_rows.append((text_val, label))

    os.makedirs("data", exist_ok=True)
    out = "data/evidence_labeled.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(out_rows)

    print(f"[ok] wrote {len(out_rows)} rows → {out}")
    session.close()

if __name__ == "__main__":
    main()
