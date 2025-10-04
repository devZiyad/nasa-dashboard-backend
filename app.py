# --- Standard library ---
import asyncio
import json
import re
import os
from collections import defaultdict

# --- Third-party libraries ---
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy.orm import Session

# --- Local modules ---
from config import Config
from db import SessionLocal
from ingest.scrape_pmc_xml import crawl_and_store
from models import init_db, Publication, Section, SectionType, Entity, Triple
from process.ai_pipeline import summarize_paper, chat_with_context, extract_entities_triples
from trends import compute_entity_trends, compute_relation_trends, compute_top_trends
from utils.text_clean import safe_truncate
from utils.nlp_clean import (
    normalize_entity,
    normalize_relation,
    is_valid_entity,
    is_valid_relation,
    normalize_confidence,
)
from vector_engine import VectorEngine


# -------------------------------------------------------------------
# App Initialization
# -------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
init_db()
load_dotenv()

VE = VectorEngine(persist=True)
VE.model.encode(["warmup"], convert_to_numpy=True)
app.logger.info("✅ VectorEngine ready")

# -------------------------------------------------------------------
# Utility / Maintenance Routes
# -------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    app.logger.info("Health check requested")
    return jsonify({"status": "ok"})


@app.route("/reset-db", methods=["POST"])
def reset_db():
    app.logger.warning("⚠️ Resetting database and FAISS index")
    db = SessionLocal()
    try:
        db.query(Section).delete()
        db.query(Publication).delete()
        db.commit()

        for path in [Config.FAISS_INDEX_PATH, Config.EMBEDDINGS_NPY_PATH, Config.ID_MAP_NPY_PATH]:
            if os.path.exists(path):
                os.remove(path)

        global VE
        VE = None

        app.logger.info("✅ Database and FAISS index reset")
        return jsonify({"status": "reset ok"})
    finally:
        db.close()


@app.route("/stats", methods=["GET"])
def stats():
    app.logger.info("📊 Stats endpoint requested")
    db: Session = SessionLocal()
    try:
        total = db.query(Publication).count()
        restricted = db.query(Publication).filter(
            Publication.xml_restricted == True).count()
        full_text = total - restricted

        journals = [j[0] for j in db.query(Publication.journal)
                    .filter(Publication.journal.isnot(None))
                    .distinct().order_by(Publication.journal).all()]

        min_year = db.query(Publication.year).filter(Publication.year.isnot(None))\
            .order_by(Publication.year.asc()).first()
        max_year = db.query(Publication.year).filter(Publication.year.isnot(None))\
            .order_by(Publication.year.desc()).first()

        app.logger.info(f"✅ Stats computed: {total} publications, {
                        restricted} restricted")
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

# -------------------------------------------------------------------
# Data Ingestion & Browsing
# -------------------------------------------------------------------


@app.route("/ingest/from-csv", methods=["POST"])
def ingest_from_csv():
    payload = request.get_json(force=True) or {}
    csv_path = payload.get("csv_path", "data/SB_publication_PMC.csv")
    limit = int(payload.get("limit", 0))
    app.logger.info(f"📥 Ingest request from CSV={csv_path}, limit={limit}")

    df = pd.read_csv(csv_path)
    urls = df["Link"].dropna().tolist()
    if limit > 0:
        urls = urls[:limit]

    res = asyncio.run(crawl_and_store(urls))
    app.logger.info(f"✅ Ingest completed: {len(res)} URLs processed")

    return jsonify({"ingest": res, "index_built": True})


@app.route("/publications", methods=["GET"])
def list_publications():
    app.logger.info("Listing publications with filters")
    page = int(request.args.get("page", 1))
    per_page = min(int(request.args.get("per_page", 20)), 100)

    db: Session = SessionLocal()
    try:
        q = db.query(Publication)

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

        total = q.count()
        rows = q.offset((page - 1) * per_page).limit(per_page).all()

        items = [{
            "id": p.id,
            "pmc_id": p.pmc_id,
            "title": p.title,
            "link": p.link,
            "journal": p.journal,
            "year": p.year,
            "xml_restricted": p.xml_restricted
        } for p in rows]

        app.logger.info(f"✅ Returned {len(items)} publications (page {
                        page}/{(total + per_page - 1) // per_page})")
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
    app.logger.info(f"Fetching paper {pub_id}")
    db: Session = SessionLocal()
    try:
        p = db.get(Publication, pub_id)
        if not p:
            app.logger.warning(f"❌ Paper {pub_id} not found")
            return jsonify({"error": "not found"}), 404
        secs = db.query(Section).filter(Section.publication_id == p.id).all()

        app.logger.info(f"✅ Paper {pub_id} fetched successfully")
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

# -------------------------------------------------------------------
# Core Features: Search & Chat
# -------------------------------------------------------------------


@app.route("/semantic-search", methods=["GET"])
def semantic_search():
    q = request.args.get("q", "").strip()
    app.logger.info(f"🔎 Semantic search query='{q}'")
    k = int(request.args.get("k", "10"))
    section = request.args.get("section", "").strip().lower() or None

    year_from = request.args.get("year_from", type=int)
    year_to = request.args.get("year_to", type=int)
    journal = request.args.get("journal")
    restricted = request.args.get("restricted")
    # fmt: off
    with_suggestions = request.args.get("suggestions", "false").lower() in ("1", "true", "yes")

    if not q:
        app.logger.warning("❌ Missing query in semantic search")
        return jsonify({"error": "missing query"}), 400

    global VE
    if VE is None:
        VE = VectorEngine(persist=True)

    # --- Semantic Search ---
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

        # --- AI Query Suggestions ---
        if with_suggestions:
            sys_prompt = (
                "You are an expert scientific search assistant. "
                "Rewrite the user's query into 3-5 alternative search queries "
                "that are database-friendly and likely to return more relevant results. "
                "Return only the queries as a JSON array."
            )
            conv_msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q}
            ]
            raw = chat_with_context(conv_msgs)
            try:
                clean = re.sub(r"^```(?:json)?|```$", "",
                               raw.strip(), flags=re.MULTILINE).strip()
                suggestions = json.loads(clean)
                if not isinstance(suggestions, list):
                    raise ValueError("Not a list")
            except Exception:
                suggestions = [line.strip("-• \n")
                               for line in raw.splitlines() if line.strip()]
                suggestions = [s for s in suggestions if len(s) > 2][:5]

            response["suggestions"] = suggestions
            app.logger.info(f"💡 Added {len(suggestions)} AI query suggestions")

        # fmt: off
        app.logger.info(f"✅ Semantic search returned {len(results)} results (fallback={fallback_used})")
        return jsonify(response)
    finally:
        db.close()


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    messages = data.get("messages", [])
    only_context = data.get("only_context", False)
    k = int(data.get("k", 5))

    user_msgs = [m for m in messages if m.get("role") == "user"]
    query = user_msgs[-1]["content"] if user_msgs else ""
    app.logger.info(f"💬 Chat request query='{query[:50]}...'")

    global VE
    if VE is None:
        VE = VectorEngine(persist=True)

    results = VE.search(query, top_k=k)

    if not results:
        app.logger.warning("❌ No relevant docs found for chat")
        return jsonify({"answer": "No relevant documents found.", "mode": "none", "citations": [], "chunks_used": 0})

    db: Session = SessionLocal()
    docs, citations = [], []
    try:
        for pub_id, kind, dist in results:
            sec = db.query(Section).filter(
                Section.publication_id == pub_id, Section.kind == kind).first()
            pub = db.get(Publication, pub_id)
            if sec and pub:
                docs.append(f"Section: {kind}\n{
                            safe_truncate(sec.text, 1500)}")
                citations.append({
                    "id": pub.id, "title": pub.title, "link": pub.link,
                    "journal": pub.journal, "year": pub.year
                })
    finally:
        db.close()

    ctx = "\n\n".join(docs)
    enough_text = len(ctx) > 50
    relevant = len(docs) > 0

    if only_context or (relevant and enough_text):
        sys_prompt = ("You are a space biology expert. Answer ONLY using the provided context. "
                      "If the answer isn't there, say 'I don't know.' Keep it concise.")
        user_prompt = f"Context:\n{ctx}\n\nQuestion: {query}"
        mode = "RAG"
    else:
        sys_prompt = ("You are a helpful science assistant. Use general knowledge. "
                      "If uncertain, say you're unsure.")
        user_prompt = query
        mode = "AI"

    history = messages[-6:]
    conv_msgs = [{"role": "system", "content": sys_prompt}] + \
        history[:-1] + [{"role": "user", "content": user_prompt}]
    answer = chat_with_context(conv_msgs)

    app.logger.info(
        f"✅ Chat answered (mode={mode}, citations={len(citations)})")
    return jsonify({
        "answer": answer, "mode": mode,
        "citations": citations, "chunks_used": len(docs)
    })

# -------------------------------------------------------------------
# Paper Processing: Summaries & Extraction
# -------------------------------------------------------------------


@app.route("/summarize/<int:pub_id>", methods=["GET"])
def summarize(pub_id: int):
    app.logger.info(f"📄 Summarize requested for pub_id={pub_id}")
    db: Session = SessionLocal()
    try:
        p = db.get(Publication, pub_id)
        if not p:
            app.logger.warning(f"❌ Summarize failed: pub_id {
                               pub_id} not found")
            return jsonify({"error": "not found"}), 404

        # ✅ Cache check
        if p.summary:
            app.logger.info(f"✅ Summarize cache hit for pub_id {pub_id}")
            return jsonify({"id": p.id, "title": p.title, "summary": p.summary})

        # ✅ Collect sections
        secs = db.query(Section).filter(Section.publication_id == p.id).all()
        abs_sec = next((s for s in secs if s.kind ==
                       SectionType.abstract), None)
        res_sec = next((s for s in secs if s.kind ==
                       SectionType.results), None)

        abstract = abs_sec.text if abs_sec else ""
        results_txt = res_sec.text if res_sec else ""

        # ✅ Fallback if empty
        if not abstract and not results_txt:
            app.logger.warning(f"Summarize: pub_id {
                               pub_id} has no abstract/results, using full text")
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

        app.logger.info(f"✅ Summarize generated & cached for pub_id {pub_id}")
        return jsonify({"id": p.id, "title": p.title, "summary": summary})
    finally:
        db.close()


@app.route("/summarize/bulk", methods=["POST"])
def summarize_bulk():
    app.logger.info("📄 Bulk summarization started")
    db: Session = SessionLocal()
    try:
        pubs = db.query(Publication).filter(
            Publication.summary.is_(None)).all()
        total = len(pubs)
        done = 0

        for pub in pubs:
            sections = db.query(Section).filter(
                Section.publication_id == pub.id).all()
            abs_text = next(
                (s.text for s in sections if s.kind == SectionType.abstract), "")
            res_text = next(
                (s.text for s in sections if s.kind == SectionType.results), "")

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
        app.logger.info(f"✅ Bulk summarization finished: {
                        done}/{total} papers summarized")
        return jsonify({"status": "ok", "summarized": done, "total": total})
    finally:
        db.close()


@app.route("/extract/<int:pub_id>", methods=["POST"])
def extract(pub_id: int):
    app.logger.info(f"🔬 Extract requested for pub_id={pub_id}")
    db: Session = SessionLocal()
    try:
        p = db.get(Publication, pub_id)
        if not p:
            app.logger.warning(f"❌ Extract failed: pub_id {pub_id} not found")
            return jsonify({"error": "not found"}), 404

        text = p.summary
        if not text:
            secs = db.query(Section).filter(
                Section.publication_id == p.id).all()
            text = " ".join(s.text for s in secs if s.text)[:8000]

        if not text.strip():
            return jsonify({"error": "no text to extract"}), 400

        # Run LLM extraction
        raw_json = extract_entities_triples(text)
        try:
            clean = re.sub(r"^```(?:json)?|```$", "",
                           raw_json.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(clean)
        except Exception as e:
            app.logger.error(
                f"❌ Failed JSON parse in extract (pub_id={pub_id}): {e}")
            return jsonify({"error": "failed to parse extraction", "raw": raw_json}), 500

        # Clear old
        db.query(Entity).filter(Entity.publication_id == p.id).delete()
        db.query(Triple).filter(Triple.publication_id == p.id).delete()

        # Save entities
        for e in parsed.get("entities", []):
            text_val = e.get("text", "")
            etype_val = e.get("type", "")
            if is_valid_entity(text_val, etype_val):
                db.add(Entity(publication_id=p.id, text=text_val, type=etype_val))

        # Save triples
        for t in parsed.get("triples", []):
            subj, rel, obj = t.get("subject", ""), t.get(
                "relation", ""), t.get("object", "")
            rel = normalize_relation(rel)  # 🔹 normalize relation here
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
        app.logger.info(f"✅ Extracted {len(parsed.get('entities', []))} entities, {
                        len(parsed.get('triples', []))} triples (pub_id={pub_id})")
        return jsonify({
            "status": "ok",
            "entities": len(parsed.get("entities", [])),
            "triples": len(parsed.get("triples", []))
        })
    finally:
        db.close()


@app.route("/extract/bulk", methods=["POST"])
def extract_bulk():
    app.logger.info("🔬 Bulk extraction started")
    db: Session = SessionLocal()
    try:
        pubs = db.query(Publication).all()
        total = len(pubs)
        done = 0

        for pub in pubs:
            if pub.entities or pub.triples:
                continue

            text = pub.summary
            if not text:
                sections = " ".join(s.text for s in pub.sections if s.text)
                text = sections[:8000]

            if not text.strip():
                app.logger.warning(f"⚠️ Skipping pub_id={pub.id}, no text")
                continue

            raw = extract_entities_triples(text)
            try:
                parsed = json.loads(raw)
            except Exception:
                app.logger.warning(f"⚠️ Failed JSON parse for pub {pub.id}")
                continue

            # Entities
            for e in parsed.get("entities", []):
                text_val = e.get("text", "")
                etype_val = e.get("type", "unknown")
                if is_valid_entity(text_val, etype_val):
                    db.add(Entity(publication_id=pub.id,
                           text=text_val, type=etype_val))

            # Triples
            for t in parsed.get("triples", []):
                subj, rel, obj = t.get("subject", ""), t.get(
                    "relation", ""), t.get("object", "")
                rel = normalize_relation(rel)  # 🔹 normalize relation here
                if subj and obj and is_valid_relation(rel):
                    db.add(Triple(
                        publication_id=pub.id,
                        subject=subj,
                        relation=rel,
                        object=obj,
                        evidence_sentence=t.get("evidence_sentence"),
                        confidence=normalize_confidence(t.get("confidence")),
                    ))

            done += 1
            if done % 5 == 0:
                db.commit()
            app.logger.info(f"✅ Extracted {done}/{total} (pub_id={pub.id})")

        db.commit()
        app.logger.info(f"✅ Bulk extraction finished: {
                        done}/{total} publications processed")
        return jsonify({"status": "ok", "processed": done, "total": total})
    finally:
        db.close()

# -------------------------------------------------------------------
# Analytics / Insights
# -------------------------------------------------------------------


@app.route("/insights/<int:pub_id>", methods=["GET"])
def insights(pub_id: int):
    app.logger.info(f"🔍 Insights requested for pub_id={pub_id}")
    db: Session = SessionLocal()
    try:
        p = db.get(Publication, pub_id)
        if not p:
            app.logger.warning(f"❌ Insights failed: pub_id {pub_id} not found")
            return jsonify({"error": "not found"}), 404

        # ✅ Cache check
        if getattr(p, "insights", None):
            app.logger.info(f"✅ Insights cache hit for pub_id={pub_id}")
            return jsonify({"id": p.id, "title": p.title, "insights": p.insights})

        # ✅ Ensure summary exists
        if not p.summary:
            secs = db.query(Section).filter(Section.publication_id == p.id).all()
            full_text = " ".join(s.text for s in secs if s.text)[:8000]
            if full_text.strip():
                p.summary = summarize_paper(p.title, full_text, "")
                db.add(p)
                db.commit()

        # ✅ Get top entities/triples
        entities = db.query(Entity).filter(Entity.publication_id == p.id).limit(3).all()
        triples = db.query(Triple).filter(Triple.publication_id == p.id).limit(3).all()

        ctx = f"Title: {p.title}\n\nSummary: {p.summary or ''}\n\n"
        if entities:
            ctx += "Entities:\n" + "\n".join(f"- {e.text} ({e.type})" for e in entities) + "\n\n"
        if triples:
            ctx += "Relations:\n" + "\n".join(
                f"- {t.subject} {t.relation} {t.object}" for t in triples
            ) + "\n\n"

        user_prompt = (
            "Provide deep insights about this study, including biological significance, "
            "potential applications, and unanswered questions. Be concise but precise.\n\n"
            f"{ctx}"
        )

        conv_msgs = [
            {"role": "system", "content": "You are a space biology expert providing insights."},
            {"role": "user", "content": user_prompt},
        ]
        insights_text = chat_with_context(conv_msgs)

        # ✅ Cache insights
        p.insights = insights_text
        db.add(p)
        db.commit()

        app.logger.info(f"✅ Insights generated & cached for pub_id={pub_id}")
        return jsonify({"id": p.id, "title": p.title, "insights": insights_text})
    finally:
        db.close()


@app.route("/trends", methods=["GET"])
def trends():
    app.logger.info("📈 Trends endpoint requested")

    # Compute trends
    entity_trends = compute_entity_trends()
    relation_trends = compute_relation_trends()

    # Top normalized items
    top_entities = compute_top_trends(entity_trends, 10, label="entities")
    top_relations = compute_top_trends(relation_trends, 5, label="relations")

    # fmt: off
    app.logger.info(
        f"✅ Trends computed: {len(top_entities)} years of entity data, {            len(top_relations)} years of relation data"
    )
    return jsonify({
        "entity_trends": top_entities,
        "relation_trends": top_relations
    })


@app.route("/gaps", methods=["GET"])
def gaps():
    app.logger.info("🕳️ Gaps detection started")
    from trends import compute_gaps
    db: Session = SessionLocal()
    try:
        min_threshold = int(request.args.get("min_threshold", 5))
        relative = float(request.args.get("relative", 0.2))
        gaps = compute_gaps(min_threshold=min_threshold, relative=relative)

        if not gaps:
            app.logger.info("✅ No gaps detected")
            return jsonify({"gaps": [], "insights": "No gaps detected."})

        gap_examples = []
        for g in gaps[:10]:
            entity = g["term"]
            pubs = (db.query(Publication)
                    .join(Entity, Entity.publication_id == Publication.id)
                    .filter(Entity.text.ilike(entity))
                    .limit(3).all())
            for pub in pubs:
                if pub.summary:
                    gap_examples.append({
                        "entity": entity, "pub_id": pub.id,
                        "title": pub.title, "summary": pub.summary[:800]
                    })

        ctx_text = "\n\n".join(
            f"Entity: {ex['entity']}\nPaper: {
                ex['title']}\nSummary: {ex['summary']}"
            for ex in gap_examples
        )
        user_prompt = ("Based on these low-coverage research areas, identify the main gaps "
                       "and suggest promising directions for future experiments.\n\n"
                       f"{ctx_text}")

        conv_msgs = [
            {"role": "system", "content": "You are a space biology expert. Be precise and concise."},
            {"role": "user", "content": user_prompt}
        ]
        insights = chat_with_context(conv_msgs)

        app.logger.info(f"✅ Gaps detection finished: {len(gaps)} gaps found")
        return jsonify({"gaps": gaps, "examples": gap_examples, "insights": insights})
    finally:
        db.close()


# -------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=(Config.FLASK_ENV != "production"), port=Config.PORT)
