# simulator/sim_engine.py
from typing import Dict, List, Tuple
from collections import defaultdict
import math
import random

from .db import connect, TRIPLES_SQL, outcome_alias_rows
# Optional: if you later add DB-driven countermeasure multipliers, this import can be enabled in db.py
try:
    from .db import cm_effects_for  # def(name) -> {outcome: multiplier}
except Exception:  # pragma: no cover
    cm_effects_for = None  # type: ignore


# ---------------------------------------------------------------------
# Relation → signed effect on outcome (+1 increase, -1 decrease, 0 neutral)
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

    # neutral / ambiguous
    "affects": 0, "modulates": 0, "associated with": 0, "showed": 0,
    "causes": 0, "alters": 0, "mediated by": 0,
    "did not cause": 0, "did not induce": 0, "used_for": 0,
    "are sites of": 0, "potentially contributes to": 0,
}

# ---------------------------------------------------------------------
# Canonical outcomes with alias lists (DB-driven with fallback)
def _load_outcome_map() -> Dict[str, List[str]]:
    rows = []
    try:
        rows = list(outcome_alias_rows())
    except Exception:
        rows = []

    if not rows:
        # fallback to your current defaults
        return {
            "bone density": [
                "bone density", "bone mass", "bmd", "bone volume fraction",
                "bv/tv", "cancellous bone", "trabecular bone", "bone thickness",
                "bone loss", "changes in bone", "rapid bone loss", "bone quality"
            ],
            "bone formation": [
                "bone formation", "osteoblast", "osteogenesis", "runx2",
                "opg", "bone lining cells"
            ],
            "bone resorption": [
                "bone resorption", "osteoclast", "trap", "rankl",
                "osteolysis", "resorption"
            ],
            "marrow adiposity": ["marrow adiposity", "mat", "adipocyte"],
        }

    out_map: Dict[str, List[str]] = {}
    for r in rows:
        oc = str(r["outcome"]).strip()
        al = str(r["alias"]).strip()
        if not oc or not al:
            continue
        out_map.setdefault(oc, []).append(al)
    for k in list(out_map.keys()):
        out_map[k].append(k)  # include canonical name itself
    return out_map


OUTCOME_CANON = _load_outcome_map()

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
        if any(k in s for k in ["loss", "decrease", "reduction", "degraded"]):
            return ("bone density", -1)
        if any(k in s for k in ["increase", "gain", "improvement"]):
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


def _edge_weight(row, scenario) -> Tuple[float, dict]:
    """
    Return (weight, factors) where factors are multiplicative components:
      base, organism, tissue, days, rad, cm
    """
    conf = float(row["confidence"] or 0.5)
    base = 0.5 + 0.5 * conf  # 0.5..1.0

    subj = (row["subject"] or "")
    obj  = (row["object"] or "")
    orgs = [o.lower() for o in (scenario.get("organism") or [])]
    tiss = [t.lower() for t in (scenario.get("tissue") or [])]

    m_org  = 1.15 if (orgs and (_softmatch(subj, orgs) or _softmatch(obj, orgs))) else 1.0
    m_tiss = 1.10 if (tiss and (_softmatch(subj, tiss) or _softmatch(obj, tiss))) else 1.0

    days = float(scenario.get("microgravity_days", 0) or 0)
    gy   = float(scenario.get("radiation_Gy", 0) or 0)
    m_days = (1.0 + min(days, 120.0) / 400.0)  # up to ~+30%
    m_rad  = (1.0 + min(gy, 2.0) / 6.0)        # up to ~+33%

    cms = scenario.get("countermeasures") or []
    m_cm = 0.9 if cms else 1.0  # simple dampener; can be replaced by DB-driven effects

    w = base * m_org * m_tiss * m_days * m_rad * m_cm
    factors = {"base": base, "organism": m_org, "tissue": m_tiss, "days": m_days, "rad": m_rad, "cm": m_cm}
    return w, factors


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _sample_normal(mu: float, sigma: float) -> float:
    # Box–Muller
    u1, u2 = random.random(), random.random()
    z = math.sqrt(-2 * math.log(max(u1, 1e-9))) * math.cos(2 * math.pi * u2)
    return mu + sigma * z


def _geom_mean(xs: List[float]) -> float:
    xs = [x for x in xs if x > 0]
    if not xs:
        return 1.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ---------------------------------------------------------------------
# Main entries

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

    # ---------- AND→OR fallback query ----------
    like_all, like_any = [], []
    args_all, args_any = [], []

    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        clause = "(subject LIKE ? OR object LIKE ? OR relation LIKE ? OR evidence_sentence LIKE ?)"
        like_all.append(clause)
        like_any.append(clause)
        like = f"%{t}%"
        args_all.extend([like, like, like, like])
        args_any.extend([like, like, like, like])

    with connect() as con:
        rows_iter = []
        if like_all:
            sql_all = TRIPLES_SQL + " WHERE " + " AND ".join(like_all) + " LIMIT 500"
            rows_iter = list(con.execute(sql_all, args_all))
        if not rows_iter and like_any:
            sql_any = TRIPLES_SQL + " WHERE " + " OR ".join(like_any) + " LIMIT 500"
            rows_iter = list(con.execute(sql_any, args_any))
        if not rows_iter:
            rows_iter = list(con.execute(TRIPLES_SQL + " LIMIT 200"))

    # ---------- Aggregate signed evidence into z-scores per outcome ----------
    outcomes = defaultdict(lambda: {
        "z": 0.0,
        "edges": [],
        "factors": {"base": [], "organism": [], "tissue": [], "days": [], "rad": [], "cm": []}
    })

    for row in rows_iter:
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
        w, f = _edge_weight(row, scenario)
        z_delta = 0.5 * sign * w
        if tiny:
            z_delta *= 0.2  # downweight ambiguous relations

        # Accumulate
        oc = outcomes[out_canon]
        oc["z"] += z_delta
        oc["edges"].append({
            "publication_id": row["publication_id"],
            "relation": row["relation"],
            "subject": subj,
            "object": obj,
            "evidence_sentence": row["evidence_sentence"],
            "confidence": float(row["confidence"] or 0.5),
            "weight": w,
            "z_delta": z_delta,
        })
        for k in oc["factors"]:
            oc["factors"][k].append(f[k])

    # ---------- Convert z → probability and compute CI via bootstrap ----------
    predictions = []
    random.seed(42)
    for name, info in outcomes.items():
        z = info["z"]

        # OPTIONAL: if you add DB-driven countermeasure multipliers per outcome
        cms = scenario.get("countermeasures") or []
        if cms and cm_effects_for:
            m_total = 1.0
            for cm in cms:
                try:
                    eff = cm_effects_for(cm)  # {outcome: multiplier}
                    m_total *= eff.get(name, 1.0)
                except Exception:
                    pass
            # gentle mapping of multiplicative m_total into z-space
            z *= (1.0 + 0.5 * math.tanh(math.log(max(1e-9, m_total))))

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

        # Explainability
        factors = info["factors"]
        f_summary = {k: round(_geom_mean(vs), 3) for k, vs in factors.items()}
        contrib_rank = sorted(
            [{"name": k, "multiplier": f_summary[k], "impact": round(abs(math.log(max(1e-9, f_summary[k]))), 3)}
             for k in f_summary],
            key=lambda x: x["impact"], reverse=True
        )
        top_edges = sorted(edges, key=lambda e: abs(e["z_delta"]), reverse=True)[:6]

        predictions.append({
            "outcome": name,
            "probability": round(p, 3),
            "direction": direction,
            "ci95": [round(lo, 3), round(hi, 3)],
            "evidence": top_edges,
            "explain": {
                "multipliers": f_summary,      # e.g., {'days': 1.22, 'rad': 1.05, ...}
                "contributors": contrib_rank,  # ranked by impact
                "n_evidence": len(edges)
            }
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


def run_curve(scenario: Dict, max_days: int = 120, step: int = 5) -> Dict:
    """
    Return probability vs days for each outcome using the same engine.
    We'll vary microgravity_days from 0..max_days (inclusive) by 'step'.
    """
    series = {}
    days_grid = list(range(0, int(max_days) + 1, int(step)))
    for d in days_grid:
        sc = dict(scenario)
        sc["microgravity_days"] = d
        res = run_simulation(sc)
        for p in res.get("predictions", []):
            series.setdefault(p["outcome"], []).append({
                "days": d,
                "prob": p["probability"],
                "ci95": p.get("ci95", [p["probability"], p["probability"]])
            })
    return {
        "scenario": scenario,
        "grid_days": days_grid,
        "series": series,
        "notes": ["Curve computed by running the same simulator across the days grid."]
    }
