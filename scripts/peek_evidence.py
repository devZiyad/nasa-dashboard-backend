# scripts/peek_evidence.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db import SessionLocal
from sqlalchemy import text

s = SessionLocal()
rows = s.execute(text("SELECT id, evidence_label, evidence_prob FROM triple LIMIT 10")).fetchall()
for r in rows: print(r)
s.close()