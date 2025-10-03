from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import asyncio
from sqlalchemy.orm import Session
from config import Config
from db import SessionLocal
from models import init_db, Publication, Section, SectionType
# 🔹 use the new XML-based scraper
from ingest.scrape_pmc_xml import crawl_and_store
from vector_engine import VectorEngine
from process.ai_pipeline import summarize_paper  # optional
from utils.text_clean import safe_truncate

app = Flask(__name__)
CORS(app)
init_db()

# Global engine (loads or builds FAISS)
VE = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ingest/from-csv", methods=["POST"])
def ingest_from_csv():
    payload = request.get_json(force=True) or {}
    csv_path = payload.get("csv_path", "data/SB_publication_PMC.csv")
    limit = int(payload.get("limit", 0))

    df = pd.read_csv(csv_path)
    urls = df["Link"].dropna().tolist()
    if limit > 0:
        urls = urls[:limit]

    import logging
    logging.getLogger("ingest").setLevel(logging.DEBUG)

    res = asyncio.run(crawl_and_store(urls))

    global VE
    VE = VectorEngine(persist=True)
    return jsonify({"ingest": res, "index_built": True})


@app.route("/publications", methods=["GET"])
def list_publications():
    db: Session = SessionLocal()
    try:
        rows = db.query(Publication).all()
        out = []
        for p in rows:
            out.append({
                "id": p.id,
                "pmc_id": p.pmc_id,
                "title": p.title,
                "link": p.link,
                "journal": p.journal,
                "year": p.year,
                "xml_restricted": p.xml_restricted
            })
        return jsonify(out)
    finally:
        db.close()


@app.route("/papers/<int:pub_id>", methods=["GET"])
def get_paper(pub_id: int):
    db: Session = SessionLocal()
    try:
        p = db.get(Publication, pub_id)
        if not p:
            return jsonify({"error": "not found"}), 404
        secs = db.query(Section).filter(Section.publication_id == p.id).all()
        return jsonify({
            "id": p.id,
            "pmc_id": p.pmc_id,
            "title": p.title,
            "link": p.link,
            "journal": p.journal,
            "year": p.year,
            "xml_restricted": p.xml_restricted,
            "sections": [
                {"id": s.id, "kind": s.kind.value,
                    "text": safe_truncate(s.text, 8000)}
                for s in secs
            ]
        })
    finally:
        db.close()


@app.route("/semantic-search", methods=["GET"])
def semantic_search():
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", "10"))
    if not q:
        return jsonify({"error": "missing query"}), 400

    global VE
    if VE is None:
        VE = VectorEngine(persist=True)

    matches = VE.search(q, top_k=k)
    db: Session = SessionLocal()
    try:
        results = []
        for pub_id, dist in matches:
            p = db.get(Publication, pub_id)
            if not p:
                continue
            results.append({
                "id": p.id,
                "title": p.title,
                "link": p.link,
                "journal": p.journal,
                "year": p.year,
                "distance": dist  # smaller = more similar
            })
        return jsonify(results)
    finally:
        db.close()


@app.route("/summarize/<int:pub_id>", methods=["GET"])
def summarize(pub_id: int):
    """Optional route: paper-level summary via OpenRouter (requires API key)."""
    db: Session = SessionLocal()
    try:
        p = db.get(Publication, pub_id)
        if not p:
            return jsonify({"error": "not found"}), 404
        abs_sec = db.query(Section).filter(
            Section.publication_id == p.id, Section.kind == SectionType.abstract).one_or_none()
        res_sec = db.query(Section).filter(
            Section.publication_id == p.id, Section.kind == SectionType.results).one_or_none()
        abstract = abs_sec.text if abs_sec else ""
        results_txt = res_sec.text if res_sec else ""
        summary = summarize_paper(p.title, abstract, results_txt)
        return jsonify({"id": p.id, "title": p.title, "summary": summary})
    finally:
        db.close()


@app.route("/reset-db", methods=["POST"])
def reset_db():
    db = SessionLocal()
    try:
        # Clear tables
        db.query(Section).delete()
        db.query(Publication).delete()
        db.commit()

        # Remove FAISS index + metadata
        import os
        for path in [Config.FAISS_INDEX_PATH, Config.EMBEDDINGS_NPY_PATH, Config.ID_MAP_NPY_PATH]:
            if os.path.exists(path):
                os.remove(path)

        # Reset in-memory VectorEngine
        global VE
        VE = None

        return jsonify({"status": "reset ok"})
    finally:
        db.close()


if __name__ == "__main__":
    app.run(debug=(Config.FLASK_ENV != "production"), port=Config.PORT)
