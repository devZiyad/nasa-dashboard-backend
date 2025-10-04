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
from dotenv import load_dotenv
from collections import defaultdict


app = Flask(__name__)
CORS(app)
init_db()
load_dotenv()

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
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    per_page = min(per_page, 100)  # safety cap

    db: Session = SessionLocal()
    try:
        q = db.query(Publication)

        # 🔹 Apply filters
        journal = request.args.get("journal")
        if journal:
            q = q.filter(Publication.journal.ilike(f"%{journal}%"))

        year_from = request.args.get("year_from", type=int)
        if year_from:
            q = q.filter(Publication.year >= year_from)

        year_to = request.args.get("year_to", type=int)
        if year_to:
            q = q.filter(Publication.year <= year_to)

        restricted = request.args.get("restricted")
        if restricted is not None:
            restricted = restricted.lower() in ("1", "true", "yes")
            q = q.filter(Publication.xml_restricted == restricted)

        # 🔹 Pagination
        total = q.count()
        rows = q.offset((page - 1) * per_page).limit(per_page).all()

        items = []
        for p in rows:
            items.append({
                "id": p.id,
                "pmc_id": p.pmc_id,
                "title": p.title,
                "link": p.link,
                "journal": p.journal,
                "year": p.year,
                "xml_restricted": p.xml_restricted
            })

        return jsonify({
            "items": items,
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page
        })
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
    section = request.args.get("section", "").strip().lower() or None

    year_from = request.args.get("year_from", type=int)
    year_to = request.args.get("year_to", type=int)
    journal = request.args.get("journal")
    restricted = request.args.get("restricted")

    if not q:
        return jsonify({"error": "missing query"}), 400

    global VE
    if VE is None:
        VE = VectorEngine(persist=True)

    # Run semantic search
    matches = VE.search(
        query=q,
        top_k=k,
        section=section,
        year_from=year_from,
        year_to=year_to,
        journal=journal,
        restricted=(restricted.lower() in ("1", "true", "yes"))
        if restricted is not None else None,
    )

    fallback_used = False
    if section and not matches:
        fallback_used = True
        matches = VE.search(q, top_k=k, section=None)

    db: Session = SessionLocal()
    try:
        grouped = defaultdict(lambda: {
            "sections": [],
            "best_dist": float("inf"),
            "title": None,
            "journal": None,
            "year": None,
            "link": None
        })

        for pub_id, kind, dist in matches:
            p = db.get(Publication, pub_id)
            if not p:
                continue

            g = grouped[p.id]
            g["title"] = p.title
            g["journal"] = p.journal
            g["year"] = p.year
            g["link"] = p.link
            g["sections"].append(kind)
            if dist < g["best_dist"]:
                g["best_dist"] = dist

        results = [
            {
                "id": pub_id,
                "title": g["title"],
                "journal": g["journal"],
                "year": g["year"],
                "link": g["link"],
                "sections": sorted(set(g["sections"])),
                "distance": g["best_dist"]
            }
            for pub_id, g in grouped.items()
        ]

        response = {"results": results}
        if fallback_used:
            response["warning"] = f"No matches found in section '{
                section}', fell back to global search."

        return jsonify(response)
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


@app.route("/stats", methods=["GET"])
def stats():
    db: Session = SessionLocal()
    try:
        total = db.query(Publication).count()
        restricted = db.query(Publication).filter(
            Publication.xml_restricted == True).count()
        full_text = total - restricted

        # distinct journals
        journals = [j[0] for j in db.query(Publication.journal)
                    .filter(Publication.journal.isnot(None))
                    .distinct().order_by(Publication.journal).all()]

        # min / max year
        #fmt: off
        min_year = db.query(Publication.year).filter(Publication.year.isnot(None)).order_by(Publication.year.asc()).first()
        #fmt: off
        max_year = db.query(Publication.year).filter(Publication.year.isnot(None)).order_by(Publication.year.desc()).first()

        return jsonify({
            "total": total,
            "restricted": restricted,
            "full_text": full_text,
            "journals": journals,
            "year_range": {
                "min": min_year[0] if min_year else None,
                "max": max_year[0] if max_year else None,
            }
        })
    finally:
        db.close()


if __name__ == "__main__":
    app.run(debug=(Config.FLASK_ENV != "production"), port=Config.PORT)
