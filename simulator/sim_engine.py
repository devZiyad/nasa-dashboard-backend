# simulator/sim_engine.py
from typing import Dict, List, Tuple
from collections import defaultdict
import math
import random

from .db import connect, TRIPLES_SQL

# ---------------------------------------------------------------------
# Relation → signed effect on outcome (+1 increase, -1 decrease, 0 neutral)
# (Covers phrasing seen in your triples: "causes decrease in", "did not cause", etc.)
RELATION_SIGN = {
    # positive / pro-effect
    "increase": +1, "increased": +1, "increases": +1,
    "induces": +1, "induced": +1, "leads to": +1,
    "upregulates": +1, "activates": +1, "stimulates": +1, "enhances": +1,
    "caused": +1,

    # negative / anti-effect
    "decrease": -1, "decreased": -1, "decreases": -1,
    "reduces": -1, "reduced": -1, "downregulates": -1,
    "inhibits": -1, "suppresses": -1, "impairs": -1,
    "causes decrease in": -1,

    # neutral / ambiguous (kept but down-weighted if needed)
    "affects": 0, "modulates": 0, "associated with": 0, "showed": 0,
    "causes": 0, "alters": 0, "mediated by": 0,
    "did not cause": 0, "did not induce": 0, "used_for": 0,
    "are sites of": 0, "potentially contributes to": 0,
}

# ---------------------------------------------------------------------
# Canonical outcomes with rich alias lists to match your triples
OUTCOME_CANON: Dict[str, List[str]] = {
    "bone density": [
        "bone density", "bone mass", "skeletal density", "bmd",
        "bone volume fraction", "bv/tv", "cancellous bone", "trabecular bone",
        "bone thickness", "loss of cancellous bone", "bone loss", "rapid bone loss",
        "changes in bone", "bone quality", "significant loss of cancellous bone",
    ],
    "bone formation": [
        "bone formation", "osteoblast activity", "osteogenesis", "runx2", "opg",
        "osteoblastic", "osteoblast", "bone lining cells", "cell cycle arrest",
    ],
    "bone resorption": [
        "bone resorption", "osteoclast activity", "trap-positive osteoclast",
        "rankl", "osteoclastic degradation", "osteocytic osteolysis", "resorption",
        "trap", "osteoclast",
    ],
    "marrow adiposity": [
        "marrow adiposity", "mat", "adipocytes", "marrow adipocyte",
    ],
}

# ---------------------------------------------------------------------
# Helpers

def _canon_outcome(term: str) -> str:
    """Map a free-text term to a canonical outcome, if any."""
    t = (term or "").lower()
    for canon, aliases in OUTCOME_CANON.items():
        if any(a in t for a in aliases):
            return canon
    return ""

def _infer_outcome_from_phrase(text: str) -> Tuple[str, int]:
    """
    Heuristic: infer (canonical_outcome, implied_sign) from a free phrase.
    Returns ("", 0) if not inferable. Sign: +1 increase, -1 decrease, 0 neutral.
    """
    s = (text or "").lower()

    # density signals
    if ("bone" in s or "bv/tv" in s or "bone volume fraction" in s
        or "cancellous" in s or "trabecular" in s or "thickness" in s
        or "bone mass" in s or "bone quality" in s):
        if "loss" in s or "decrease" in s or "reduction" in s or "degraded" in s:
            return ("bone density", -1)
        if "increase" in s or "gain" in s or "improvement" in s:
            return ("bone density", +1)

    # resorption signals
    if any(k in s for k in ["osteoclast", "resorption", "trap-positive", "rankl", "osteolysis"]):
        if any(k in s for k in ["decrease", "reduced", "inhibit", "suppressed", "lower"]):
            return ("bone resorption", -1)
        return ("bone resorption", +1)

    # formation signals
    if any(k in s for k in ["osteoblast", "osteogenesis", "runx2", "bone lining cells"]):
        if any(k in s for k in ["arrest", "inhibit", "decrease", "reduced", "lower"]):
            return ("bone formation", -1)
        return ("bone formation", +1)

    # marrow adiposity
    if any(k in s for k in ["marrow adiposity", "mat", "adipocyte"]):
        if any(k in s for k in ["decrease", "reduced", "lower"]):
            return ("marrow adiposity", -1)
        return ("marrow adiposity", +1)

    return ("", 0)

def _softmatch(hay: str, needles: List[str]) -> bool:
    H = (hay or "").lower()
    return any((n or "").lower() in H for n in needles if n)

def _edge_weight(row, scenario) -> float:
    """
    Compute an evidence edge weight using triple confidence,
    organism/tissue matching, exposure (days/Gy), and countermeasures.
    """
    conf = float(row["confidence"] or 0.5)
    w = 0.5 + 0.5 * conf  # 0.5..1.0 base on confidence

    subj = (row["subject"] or "")
    obj  = (row["object"] or "")
    orgs = [o.lower() for o in (scenario.get("organism") or [])]
    tiss = [t.lower() for t in (scenario.get("tissue") or [])]

    if orgs and (_softmatch(subj, orgs) or _softmatch(obj, orgs)):
        w *= 1.15
    if tiss and (_softmatch(subj, tiss) or _softmatch(obj, tiss)):
        w *= 1.10

    days = float(scenario.get("microgravity_days", 0) or 0)
    gy   = float(scenario.get("radiation_Gy", 0) or 0)
    w *= (1.0 + min(days, 120.0) / 400.0)  # up to ~+30% at 120 days
    w *= (1.0 + min(gy, 2.0) / 6.0)        # up to ~+33% at 2 Gy

    cms = scenario.get("countermeasures") or []
    if cms:
        w *= 0.9  # conservative dampening for countermeasure presence
    return w

def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _sample_normal(mu: float, sigma: float) -> float:
    # Box–Muller transform
    u1, u2 = random.random(), random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + sigma * z

# ---------------------------------------------------------------------
# Main entry

def run_simulation(scenario: Dict) -> Dict:
    """
    Run a scenario-based simulation.

    scenario example:
    {
      "question": "microgravity bone",
      "organism": ["mouse", "mice"],
      "tissue": ["bone", "calvaria"],
      "microgravity_days": 30,
      "radiation_Gy": 0.2,
      "countermeasures": ["treadmill"]
    }
    """

    # Build search terms for triple retrieval
    terms: List[str] = []
    q = (scenario.get("question") or "").strip()
    if q:
        terms.extend(q.split())
    for key in ("organism", "tissue"):
        for v in (scenario.get(key) or []):
            terms.append(v)
    terms.append("microgravity")
    if (scenario.get("radiation_Gy") or 0) > 0:
        terms.append("radiation")

    like_where, args = [], []
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        like_where.append("(subject LIKE ? OR object LIKE ? OR relation LIKE ? OR evidence_sentence LIKE ?)")
        like = f"%{t}%"
        args.extend([like, like, like, like])

    sql = TRIPLES_SQL
    if like_where:
        sql += " WHERE " + " AND ".join(like_where)
    sql += " LIMIT 500"

    # Aggregate signed evidence into z-scores per canonical outcome
    outcomes = defaultdict(lambda: {"z": 0.0, "edges": []})
    with connect() as con:
        for row in con.execute(sql, args):
            rel = (row["relation"] or "").lower().strip()

            # Determine outcome
            subj = (row["subject"] or "")
            obj  = (row["object"]  or "")
            out_canon = _canon_outcome(subj) or _canon_outcome(obj)
            implied = 0
            if not out_canon:
                out_canon, implied = _infer_outcome_from_phrase(subj)
            if not out_canon:
                out_canon, implied = _infer_outcome_from_phrase(obj)
            if not out_canon:
                continue

            # Determine sign
            sign = RELATION_SIGN.get(rel, 0)
            tiny = False
            if implied != 0 and sign == 0:
                sign = implied
            if sign == 0:
                # still ambiguous; include as tiny positive so evidence shows up
                sign = +1
                tiny = True

            # Weighting & z-score contribution
            w = _edge_weight(row, scenario)
            z_delta = 0.5 * sign * w
            if tiny:
                z_delta *= 0.2  # downweight ambiguous relations

            # Accumulate
            outcomes[out_canon]["z"] += z_delta
            outcomes[out_canon]["edges"].append({
                "publication_id": row["publication_id"],
                "relation": row["relation"],
                "subject": subj,
                "object": obj,
                "evidence_sentence": row["evidence_sentence"],
                "confidence": float(row["confidence"] or 0.5),
                "weight": w,
                "z_delta": z_delta,
            })

    # Convert z → probability and compute CI via bootstrap
    predictions = []
    random.seed(42)
    for name, info in outcomes.items():
        z = info["z"]
        p = _logistic(z)

        edges = info["edges"]
        if edges:
            n = len(edges)
            samples: List[float] = []
            for _ in range(200):  # small bootstrap
                z_b = 0.0
                for __ in range(n):
                    e = edges[random.randrange(n)]
                    z_b += _sample_normal(e["z_delta"], sigma=0.05 * abs(e["z_delta"]) + 0.01)
                samples.append(_logistic(z_b))
            samples.sort()
            lo = samples[int(0.025 * len(samples))]
            hi = samples[int(0.975 * len(samples))]
        else:
            lo = hi = p

        direction = "increase" if p >= 0.55 else ("decrease" if p <= 0.45 else "no_change")

        top_edges = sorted(edges, key=lambda e: abs(e["z_delta"]), reverse=True)[:6]
        predictions.append({
            "outcome": name,
            "probability": round(p, 3),
            "direction": direction,
            "ci95": [round(lo, 3), round(hi, 3)],
            "evidence": top_edges,
        })

    if not predictions:
        return {
            "scenario": scenario,
            "predictions": [],
            "notes": ["No matching evidence in triples for this scenario. Try broader tissue/outcome terms."],
        }

    predictions.sort(key=lambda x: abs(0.5 - x["probability"]), reverse=True)
    return {
        "scenario": scenario,
        "predictions": predictions,
        "notes": [
            "Probabilities are derived from signed evidence aggregation over triples with logistic mapping.",
            "Confidence intervals estimated via bootstrap over evidence edges (n=200).",
        ],
    }
