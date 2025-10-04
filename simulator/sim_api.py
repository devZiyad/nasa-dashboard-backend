from flask import Blueprint, request, jsonify
from flask_cors import CORS
from .retriever import search, ensure_index
from .simulator import simulate
from .graph_api import graph_query
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
    res = run_simulation(payload)
    return jsonify(res)


@sim_bp.get("/schema")
def api_schema():
    """
    Returns the simulator's input knobs and known outcomes so the UI
    can build forms/sliders without hardcoding.
    """
    schema = {
        "inputs": {
            "question": {"type": "string", "example": "microgravity bone"},
            "organism": {"type": "list[string]", "examples": ["mouse", "mice", "rat", "human"]},
            "tissue": {"type": "list[string]", "examples": ["bone", "calvaria", "femur", "osteoblast", "osteoclast"]},
            "microgravity_days": {"type": "number", "min": 0, "max": 180, "default": 30},
            "radiation_Gy": {"type": "number", "min": 0, "max": 2.0, "default": 0.0},
            "countermeasures": {"type": "list[string]", "examples": ["treadmill", "bisphosphonate", "vitamin D"]},
        },
        "outcomes": [
            "bone density",
            "bone formation",
            "bone resorption",
            "marrow adiposity"
        ],
        "notes": [
            "Predictions are derived from signed evidence aggregation over triples with logistic mapping.",
            "Uncertainty (95% CI) estimated via bootstrap on evidence edges."
        ]
    }
    return jsonify(schema)
