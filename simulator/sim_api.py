from flask import Blueprint, request, jsonify
from flask_cors import CORS
from .retriever import search, ensure_index
from .simulator import simulate
from .graph_api import graph_query
from .sim_engine import run_simulation
from .db import distinct_entities_by_type, distinct_interventions, outcome_alias_rows
from .sim_engine import run_curve
from flask import Blueprint, request, jsonify
from .sim_engine import run_simulation
from flask import Blueprint, request, jsonify
from .sim_engine import run_simulation, run_curve 
from .sim_engine import run_simulation

sim_bp = Blueprint("sim", __name__)
CORS(sim_bp)

@sim_bp.get("/health")
def health():
    return jsonify({"ok": True})

@sim_bp.get("/stats")
def stats():
    idx = ensure_index()
    return jsonify({
        "documents": len(idx.docs),
        "indexed_chars": sum(len(d["text"]) for d in idx.docs),
    })

@sim_bp.post("/search")
def api_search():
    payload = request.get_json(force=True) or {}
    q = payload.get("q", "")
    k = int(payload.get("k", 10))
    filters = payload.get("filters") or {}
    results = search(q, k=k, filters=filters)
    return jsonify({"query": q, "results": results})

@sim_bp.post("/simulate")
def api_simulate():
    payload = request.get_json(force=True) or {}
    question = payload.get("question", "")
    top_k = int(payload.get("top_k", 8))
    knobs = payload.get("knobs") or {}
    filters = payload.get("filters") or {}

    # primary retrieval
    context = search(question, k=top_k, filters=filters)

    # fallback only if nothing found
    if not context:
        try:
            from .retriever import fallback_sql_like
            context = fallback_sql_like(question, k=top_k)
        except Exception:
            context = []

    sim = simulate(question, context, knobs=knobs)
    return jsonify(sim)

@sim_bp.post("/graph")
def api_graph():
    payload = request.get_json(force=True) or {}
    q = payload.get("q", "")
    limit = int(payload.get("limit", 200))
    return jsonify(graph_query(q, limit=limit))



@sim_bp.post("/run")
def api_run():
    """
    Run a scenario-based simulation.
    Body example:
    {
      "question": "microgravity effects on bone",
      "organism": ["mouse"],
      "tissue": ["bone","calvaria"],
      "microgravity_days": 30,
      "radiation_Gy": 0.0,
      "countermeasures": ["treadmill"]
    }
    """
    payload = request.get_json(force=True) or {}
    scenario = _sanitize_scenario(payload)  # ✅ sanitize here
    return jsonify(run_simulation(scenario))

@sim_bp.get("/schema")
def api_schema():
    """
    Build the simulator's input schema dynamically from the DB.
    Falls back gracefully if some entity types are missing.
    """
    organisms = distinct_entities_by_type("organism", limit=300)
    tissues   = distinct_entities_by_type("tissue/system", limit=300)
    if not tissues:  # some datasets use slightly different label
        tissues = distinct_entities_by_type("tissue", limit=300)

    cms = distinct_interventions(limit=300)

    # outcomes: use your current hard-coded set as authoritative,
    # but allow optional enrichment from an outcome_alias table if present.
    base_outcomes = [
        "bone density",
        "bone formation",
        "bone resorption",
        "marrow adiposity",
    ]
    aliases = outcome_alias_rows()
    outcome_alias_map = {}
    for row in aliases:
        outcome_alias_map.setdefault(row["outcome"], []).append(row["alias"])

    schema = {
        "inputs": {
            "question": {"type": "string", "example": "microgravity bone"},
            "organism": {"type": "list[string]", "values": organisms[:300]},
            "tissue": {"type": "list[string]", "values": tissues[:300]},
            "microgravity_days": {"type": "number", "min": 0, "max": 180, "default": 30},
            "radiation_Gy": {"type": "number", "min": 0, "max": 2.0, "default": 0.0},
            "countermeasures": {"type": "list[string]", "values": cms[:300]},
        },
        "outcomes": base_outcomes,
        "outcome_aliases": outcome_alias_map,  # empty {} if no table present
        "notes": [
            "Organisms/tissues/countermeasures are discovered from the database (entity table).",
            "If you add an 'outcome_alias' table, outcome labels/aliases will be discovered as well.",
        ],
    }
    return jsonify(schema)

@sim_bp.post("/curve")
def api_curve():
    """
    Body:
    {
      "scenario": {...},      // same fields as /sim/run
      "max_days": 120,
      "step": 5
    }
    """
    payload = request.get_json(force=True) or {}
    scenario = _sanitize_scenario(payload.get("scenario") or {})
    max_days = int(payload.get("max_days", 120))
    step = int(payload.get("step", 5))
    return jsonify(run_curve(scenario, max_days=max_days, step=step))


@sim_bp.post("/compare")
def api_compare():
    """
    Compare two scenarios and return per-outcome deltas.
    Body:
    {
      "baseline": {...},
      "variant":  {...}
    }
    """
    payload = request.get_json(force=True) or {}
    base = payload.get("baseline") or {}
    var  = payload.get("variant")  or {}

    rA = run_simulation(base)
    rB = run_simulation(var)

    A = {p["outcome"]: p for p in rA.get("predictions", [])}
    B = {p["outcome"]: p for p in rB.get("predictions", [])}

    keys = sorted(set(A) | set(B))
    diffs = []
    for k in keys:
        a = A.get(k, {"probability": 0.5, "ci95":[0.5,0.5]})
        b = B.get(k, {"probability": 0.5, "ci95":[0.5,0.5]})
        diffs.append({
            "outcome": k,
            "baseline_prob": a["probability"],
            "variant_prob":  b["probability"],
            "delta": round(b["probability"] - a["probability"], 3),
            "baseline_ci95": a.get("ci95", [a["probability"], a["probability"]]),
            "variant_ci95":  b.get("ci95", [b["probability"], b["probability"]]),
        })

    return jsonify({
        "baseline": rA.get("scenario"),
        "variant":  rB.get("scenario"),
        "diffs": sorted(diffs, key=lambda x: abs(x["delta"]), reverse=True)
    })

@sim_bp.post("/explain")
def api_explain():
    """
    Same input as /sim/run. Returns predictions with an 'explain' block per outcome.
    """
    scenario = request.get_json(force=True) or {}
    return jsonify(run_simulation(scenario))



@sim_bp.post("/compare")
def api_compare():
    """
    Body:
    {
      "baseline": {...},   # same schema as /sim/run
      "variant":  {...}
    }
    """
    payload = request.get_json(force=True) or {}
    baseline = _sanitize_scenario(payload.get("baseline") or {})
    variant = _sanitize_scenario(payload.get("variant") or {})

    base_result = run_simulation(baseline)
    variant_result = run_simulation(variant)

    return jsonify({
        "baseline": base_result,
        "variant": variant_result
    })

    # minimal validation (see sanitizer below)
    if not isinstance(baseline, dict) or not isinstance(variant, dict):
        return jsonify({"error": "baseline and variant must be objects"}), 400

    from .sim_engine import compare_scenarios
    return jsonify(compare_scenarios(_sanitize_scenario(baseline),
                                     _sanitize_scenario(variant)))


def _sanitize_scenario(sc: dict) -> dict:
    if not isinstance(sc, dict):
        return {}
    out = dict(sc)

    # normalize list fields
    for key in ("organism", "tissue", "countermeasures"):
        vals = out.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            vals = []
        out[key] = [str(v).strip() for v in vals if str(v).strip()]

    # numeric clamps
    def _num(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    days = _num(out.get("microgravity_days", 0))
    out["microgravity_days"] = max(0.0, min(days, 365.0))

    gy = _num(out.get("radiation_Gy", 0))
    out["radiation_Gy"] = max(0.0, min(gy, 10.0))

    # question safe default
    q = out.get("question") or ""
    out["question"] = str(q)[:200]

    return out
