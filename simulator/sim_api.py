from flask import Blueprint, request, jsonify
from flask_cors import CORS
from .retriever import search, ensure_index
from .simulator import simulate
from .graph_api import graph_query

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
    context = search(question, k=top_k, filters=filters)
    return jsonify(simulate(question, context, knobs=knobs))

@sim_bp.post("/graph")
def api_graph():
    payload = request.get_json(force=True) or {}
    q = payload.get("q", "")
    limit = int(payload.get("limit", 200))
    return jsonify(graph_query(q, limit=limit))
