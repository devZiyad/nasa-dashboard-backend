from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import asyncio
import json
import re
import spacy
from sqlalchemy.orm import Session
from config import Config
from db import SessionLocal
from models import init_db, Publication, Section, SectionType, Entity, Triple
# 🔹 use the new XML-based scraper
from ingest.scrape_pmc_xml import crawl_and_store
from vector_engine import VectorEngine
from process.ai_pipeline import summarize_paper, chat_with_context, extract_entities_triples
from utils.text_clean import safe_truncate
from dotenv import load_dotenv
from collections import defaultdict
from trends import compute_entity_trends, compute_relation_trends, compute_top_trends
from simulator.sim_api import sim_bp
from db import connect
from ml.retriever_infer import build_index
from flask import request, jsonify
from ml.retriever_infer import search


app = Flask(__name__)
app.register_blueprint(sim_bp, url_prefix="/sim")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
init_db()
load_dotenv()

nlp = spacy.load("en_core_web_sm")

global VE
VE = None


VE = VectorEngine(persist=True)
VE.model.encode(["warmup"], convert_to_numpy=True)
app.logger.info("✅ VectorEngine ready")


CONFIDENCE_MAP = {
    "high": 0.95,
    "medium": 0.6,
    "low": 0.3
}


def is_valid_entity(text: str, etype: str | None = None) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 3:
        return False
    doc = nlp(t)
    if not any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc):
        return False
    if all(tok.is_stop for tok in doc):
        return False
    if etype and etype.lower() not in {
        "organism", "gene", "protein", "tissue", "condition", "outcome"
    }:
        return False
    return True


def is_valid_relation(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 3:
        return False
    doc = nlp(t)
    verbs = [tok for tok in doc if tok.pos_ == "VERB"]
    nouns = [tok for tok in doc if tok.pos_ in {"NOUN", "PROPN"}]
    if not verbs or not nouns:
        return False
    if len(doc) == 1 and doc[0].pos_ == "VERB":  # reject bare verbs
        return False
    return True


def normalize_confidence(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_lower = val.strip().lower()
        if val_lower in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[val_lower]
        try:
            return float(val)
        except ValueError:
            return None
    return None


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

    res = asyncio.run(crawl_and_store(urls))

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
            # fmt: off
            response["warning"] = f"No matches found in section '{section}', fell back to global search."

        return jsonify(response)
    finally:
        db.close()


@app.route("/summarize/<int:pub_id>", methods=["GET"])
def summarize(pub_id: int):
    """Paper-level summary via OpenRouter (cached in DB)."""
    db: Session = SessionLocal()
    try:
        p = db.get(Publication, pub_id)
        if not p:
            app.logger.warning(f"Summarize: pub_id {pub_id} not found")
            return jsonify({"error": "not found"}), 404

        # ✅ Check cache first
        if p.summary:
            #fmt: off
            app.logger.info(f"Summarize: pub_id {pub_id} → using cached summary")
            return jsonify({"id": p.id, "title": p.title, "summary": p.summary})

        # ✅ Collect sections
        secs = db.query(Section).filter(Section.publication_id == p.id).all()
        abs_sec = next((s for s in secs if s.kind ==
                       SectionType.abstract), None)
        res_sec = next((s for s in secs if s.kind ==
                       SectionType.results), None)

        abstract = abs_sec.text if abs_sec else ""
        results_txt = res_sec.text if res_sec else ""

        # ✅ If empty → fallback to all text
        if not abstract and not results_txt:
            # fmt: off
            app.logger.warning(f"Summarize: pub_id {pub_id} has no abstract/results, falling back to full text")
            full_text = " ".join(s.text for s in secs if s.text)[:8000]
            if not full_text.strip():
                return jsonify({"error": "no content to summarize"}), 400
            summary = summarize_paper(p.title, full_text, "")
        else:
            summary = summarize_paper(p.title, abstract, results_txt)

        # ✅ Save back to DB
        p.summary = summary
        db.add(p)
        db.commit()

        # fmt: off
        app.logger.info(f"Summarize: pub_id {pub_id} summary generated & cached")
        return jsonify({"id": p.id, "title": p.title, "summary": summary})
    finally:
        db.close()


@app.route("/summarize/bulk", methods=["POST"])
def summarize_bulk():
    """Summarize all publications missing a summary."""
    db: Session = SessionLocal()
    try:
        pubs = db.query(Publication).filter(Publication.summary.is_(None)).all()
        total = len(pubs)
        done = 0

        for pub in pubs:
            # Collect sections
            sections = db.query(Section).filter(Section.publication_id == pub.id).all()

            abs_text = next((s.text for s in sections if s.kind == SectionType.abstract), "")
            res_text = next((s.text for s in sections if s.kind == SectionType.results), "")

            # If no abstract/results, fall back to all sections
            if not abs_text and not res_text:
                full_text = " ".join(s.text for s in sections if s.text)[:8000]
                if not full_text.strip():
                    continue
                summary = summarize_paper(pub.title, full_text, "")
            else:
                summary = summarize_paper(pub.title, abs_text, res_text)

            if not summary:
                continue

            pub.summary = summary
            done += 1

            if done % 5 == 0:
                db.commit()
            app.logger.info(f"✅ Summarized {done}/{total} (pub_id={pub.id})")

        db.commit()
        return jsonify({"status": "ok", "summarized": done, "total": total})
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

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    messages = data.get("messages", [])
    only_context = data.get("only_context", False)
    k = int(data.get("k", 5))

    # Extract last user query
    user_msgs = [m for m in messages if m.get("role") == "user"]
    query = user_msgs[-1]["content"] if user_msgs else ""

    global VE
    if VE is None:
        VE = VectorEngine(persist=True)

    # Run semantic search from FAISS
    results = VE.search(query, top_k=k)

    if not results:
        return jsonify({"answer": "No relevant documents found.", "mode": "none", "citations": [], "chunks_used": 0})


    # Build context with real text
    db: Session = SessionLocal()
    docs, citations = [], []
    try:
        for pub_id, kind, dist in results:
            sec = (
                db.query(Section)
                .filter(Section.publication_id == pub_id, Section.kind == kind)
                .first()
            )
            pub = db.get(Publication, pub_id)
            if sec and pub:
                docs.append(f"Section: {kind}\n{safe_truncate(sec.text, 1500)}")
                citations.append({
                    "id": pub.id,
                    "title": pub.title,
                    "link": pub.link,
                    "journal": pub.journal,
                    "year": pub.year
                })
    finally:
        db.close()

    ctx = "\n\n".join(docs)

    # Check context quality
    enough_text = len(ctx) > 50
    relevant = len(docs) > 0

    if only_context or (relevant and enough_text):
        sys_prompt = (
            "You are a space biology expert. Answer ONLY using the provided context. "
            "If the answer isn't there, say 'I don't know.' Keep it concise."
        )
        user_prompt = f"Context:\n{ctx}\n\nQuestion: {query}"
        mode = "RAG"
    else:
        sys_prompt = (
            "You are a helpful science assistant. Use general knowledge. "
            "If uncertain, say you're unsure."
        )
        user_prompt = query
        mode = "AI"

    # Keep short history
    history = messages[-6:]
    conv_msgs = (
        [{"role": "system", "content": sys_prompt}]
        + history[:-1]
        + [{"role": "user", "content": user_prompt}]
    )

    answer = chat_with_context(conv_msgs)

    return jsonify({
        "answer": answer,
        "mode": mode,
        "citations": citations,
        "chunks_used": len(docs)
    })

@app.route("/extract/<int:pub_id>", methods=["POST"])
def extract(pub_id: int):
    db: Session = SessionLocal()
    try:
        p = db.get(Publication, pub_id)
        if not p:
            return jsonify({"error": "not found"}), 404

        # Get text to extract from
        text = p.summary
        if not text:
            secs = db.query(Section).filter(Section.publication_id == p.id).all()
            text = " ".join(s.text for s in secs if s.text)[:8000]

        if not text.strip():
            return jsonify({"error": "no text to extract"}), 400

        # Run LLM extraction
        raw_json = extract_entities_triples(text)

        try:
            clean = re.sub(r"^```(?:json)?|```$", "", raw_json.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(clean)
        except Exception:
            return jsonify({"error": "failed to parse extraction", "raw": raw_json}), 500

        # Clear old entities/triples
        db.query(Entity).filter(Entity.publication_id == p.id).delete()
        db.query(Triple).filter(Triple.publication_id == p.id).delete()

        # Save new ones
        for e in parsed.get("entities", []):
            text_val = e.get("text", "")
            etype_val = e.get("type", "")
            if is_valid_entity(text_val, etype_val):
                db.add(Entity(
                    publication_id=p.id,
                    text=text_val,
                    type=etype_val,
                ))

        for t in parsed.get("triples", []):
            subj, rel, obj = t.get("subject", ""), t.get("relation", ""), t.get("object", "")
            if subj and obj and is_valid_relation(rel):
                db.add(Triple(
                    publication_id=p.id,
                    subject=subj,
                    relation=rel,
                    object=obj,
                    evidence_sentence=t.get("evidence_sentence"),
                    confidence=normalize_confidence(t.get("confidence")),
                ))

        db.commit()
        return jsonify({"status": "ok", "entities": len(parsed.get("entities", [])), "triples": len(parsed.get("triples", []))})
    finally:
        db.close()

@app.route("/extract/bulk", methods=["POST"])
def extract_bulk():
    """Extract entities and triples for all publications missing them."""
    db: Session = SessionLocal()
    try:
        pubs = db.query(Publication).all()
        total = len(pubs)
        done = 0

        for pub in pubs:
            # Skip if already has entities/triples
            if pub.entities or pub.triples:
                continue

            # Prefer summary, else join sections
            text = pub.summary
            if not text:
                sections = " ".join(s.text for s in pub.sections if s.text)
                text = sections[:8000]  # safety cap

            if not text.strip():
                continue

            # Run LLM extraction
            raw = extract_entities_triples(text)
            try:
                parsed = json.loads(raw)
            except Exception:
                app.logger.warning(f"⚠️ Failed JSON parse for pub {pub.id}")
                continue

            entities = parsed.get("entities", [])
            triples = parsed.get("triples", [])

            for e in entities:
                text_val = e.get("text", "")
                etype_val = e.get("type", "unknown")
                if is_valid_entity(text_val, etype_val):
                    db.add(Entity(
                        publication_id=pub.id,
                        text=text_val,
                        type=etype_val
                    ))

            for t in triples:
                subj, rel, obj = t.get("subject", ""), t.get("relation", ""), t.get("object", "")
                if subj and obj and is_valid_relation(rel):
                    db.add(Triple(
                        publication_id=pub.id,
                        subject=subj,
                        relation=rel,
                        object=obj,
                        evidence_sentence=t.get("evidence_sentence"),
                        confidence=normalize_confidence(t.get("confidence"))
                    ))

            done += 1
            if done % 5 == 0:
                db.commit()
            app.logger.info(f"✅ Extracted {done}/{total} (pub_id={pub.id})")

        db.commit()
        return jsonify({"status": "ok", "processed": done, "total": total})
    finally:
        db.close()

@app.route("/trends", methods=["GET"])
def trends():
    entity_trends = compute_entity_trends()
    relation_trends = compute_relation_trends()

    top_entities = compute_top_trends(entity_trends, 10)
    top_relations = compute_top_trends(relation_trends, 5)

    return jsonify({"entity_trends": top_entities, "relation_trends": top_relations})

def load_texts_from_db():
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, COALESCE(abstract, title) FROM publications")
        rows = cur.fetchall()
        ids = [r[0] for r in rows]
        texts = [r[1] or "" for r in rows]
    return texts, ids

texts, ids = load_texts_from_db()
build_index(texts, ids)

@app.post("/search")
def search_api():
    q = (request.get_json() or {}).get("query", "")
    k = int((request.get_json() or {}).get("k", 10))
    hits = search(q, k=k)
    return jsonify({"ok": True, "hits": [{"id": i, "score": s} for i,s in hits]})
@app.route("/gaps", methods=["GET"])
def gaps():
    """Detect knowledge gaps (low-coverage entities) and summarize them."""
    from trends import compute_gaps
    db: Session = SessionLocal()
    try:
        # Step 1: detect gaps
        min_threshold = int(request.args.get("min_threshold", 5))
        relative = float(request.args.get("relative", 0.2))
        gaps = compute_gaps(min_threshold=min_threshold, relative=relative)

        if not gaps:
            return jsonify({"gaps": [], "insights": "No gaps detected."})

        # Step 2: get representative summaries for each gap entity
        gap_examples = []
        for g in gaps[:10]:  # limit to top 10 rare entities
            entity = g["term"]
            pubs = (
                db.query(Publication)
                .join(Entity, Entity.publication_id == Publication.id)
                .filter(Entity.text.ilike(entity))
                .limit(3)
                .all()
            )
            for pub in pubs:
                if pub.summary:
                    gap_examples.append({
                        "entity": entity,
                        "pub_id": pub.id,
                        "title": pub.title,
                        "summary": pub.summary[:800]  # safety truncate
                    })

        # Step 3: generate AI insight
        ctx_text = "\n\n".join(
            f"Entity: {ex['entity']}\nPaper: {ex['title']}\nSummary: {ex['summary']}"
            for ex in gap_examples
        )
        user_prompt = (
            "Based on these low-coverage research areas, identify the main gaps "
            "and suggest promising directions for future experiments.\n\n"
            f"{ctx_text}"
        )

        conv_msgs = [
            {"role": "system", "content": "You are a space biology expert. Be precise and concise."},
            {"role": "user", "content": user_prompt}
        ]
        insights = chat_with_context(conv_msgs)

        return jsonify({
            "gaps": gaps,
            "examples": gap_examples,
            "insights": insights
        })
    finally:
        db.close()

if __name__ == "__main__":
    app.run(debug=(Config.FLASK_ENV != "production"), port=Config.PORT)
