# process/deep_research.py
import json
import numpy as np
from sqlalchemy.orm import Session
from db import SessionLocal
from models import Publication, Entity, Triple
from vector_engine import VectorEngine
from process.ai_pipeline import _llm
from trends import compute_entity_trends
from utils.text_clean import safe_truncate


def _normalize_score(distance: float) -> float:
    """
    Convert FAISS distance (0 = identical) to a 0–100 originality score.
    """
    if distance < 0:
        return 100.0
    return max(0.0, min(100.0, (1 - min(distance / 1.2, 1.0)) * 100))


def _compare_entities(section_text: str, db: Session) -> tuple[list[str], float]:
    """
    Compare extracted entities in the input section to the DB to estimate conceptual novelty.
    """
    words = [w.strip().lower() for w in section_text.split() if len(w) > 4]
    all_entities = [e.text.lower() for e in db.query(Entity.text).all()]
    overlap = len(set(words) & set(all_entities))
    ratio = overlap / max(len(words), 1)
    conceptual_originality = (1 - min(ratio * 10, 1)) * 100
    matched_entities = list(set(words) & set(all_entities))
    return matched_entities[:10], conceptual_originality


def _get_trend_penalty(entities: list[str]) -> float:
    """
    Lower novelty if entities are trending heavily (common in recent years).
    """
    trends = compute_entity_trends()
    latest_years = sorted(trends.keys())[-3:] if trends else []
    freq = 0
    total = 0
    for y in latest_years:
        for e in entities:
            freq += trends[y].get(e, 0)
            total += 1
    if total == 0:
        return 0
    density = freq / total
    penalty = min(density * 5, 20)  # up to -20% novelty
    return penalty


def analyze_research_paper(title: str, sections: dict[str, str], top_k: int = 5):
    """
    Deep originality and feedback analysis of a research paper.
    """
    db: Session = SessionLocal()
    ve = VectorEngine(persist=True)

    results = {}
    novelty_scores = []
    insights_cache = []
    try:
        for sec_name, text in sections.items():
            if not text or len(text.strip()) < 100:
                continue

            # Step 1 — semantic search
            matches = ve.search(text, top_k=top_k, section=sec_name.lower())
            avg_dist = np.mean([m[2] for m in matches]) if matches else 1.0
            semantic_novelty = _normalize_score(avg_dist)

            # Step 2 — conceptual overlap
            matched_entities, conceptual_novelty = _compare_entities(text, db)

            # Step 3 — adjust for recent popularity
            trend_penalty = _get_trend_penalty(matched_entities)
            final_novelty = max(0, (semantic_novelty + conceptual_novelty) / 2 - trend_penalty)

            # Step 4 — sample similar works for context
            related = []
            for pub_id, section_kind, dist in matches[:3]:
                p = db.get(Publication, pub_id)
                if not p:
                    continue
                related.append({
                    "title": p.title,
                    "journal": p.journal,
                    "year": p.year,
                    "distance": round(dist, 3),
                    "summary": safe_truncate(p.summary or "", 500)
                })
            insights_cache.extend([r["title"] for r in related])

            # Step 5 — LLM feedback generation
            sys_prompt = (
                "You are a scientific reviewer AI. "
                "Assess the provided section for novelty, originality, and potential improvement. "
                "Be specific about what aspects are unique, what overlaps with existing work, "
                "and suggest how to strengthen the section."
            )
            user_prompt = f"""
            Research Title: {title}
            Section: {sec_name}
            Section Text:
            {safe_truncate(text, 4000)}

            Similar Prior Work (context):
            {json.dumps(related, indent=2)}

            Key entities found in your section:
            {', '.join(matched_entities)}

            Provide:
            - Novelty assessment (0–100 scale)
            - Overlap commentary
            - Strengths
            - Weaknesses
            - Suggestions for improvement
            - Recommended related topics
            """

            feedback = _llm("openai/gpt-4.1-mini", sys_prompt, user_prompt, temperature=0.4)

            results[sec_name] = {
                "semantic_novelty": round(semantic_novelty, 2),
                "conceptual_novelty": round(conceptual_novelty, 2),
                "trend_penalty": round(trend_penalty, 2),
                "final_novelty": round(final_novelty, 2),
                "similar_works": related,
                "matched_entities": matched_entities,
                "feedback": feedback.strip(),
            }

            novelty_scores.append(final_novelty)

        overall_score = round(np.mean(novelty_scores), 2) if novelty_scores else 0

        # Step 6 — Global synthesis (meta-analysis)
        sys_prompt = "You are an expert scientific analyst summarizing originality and improvement opportunities across a paper."
        user_prompt = f"""
        Title: {title}
        Section results:
        {json.dumps(results, indent=2)}

        Summarize:
        - Overall originality
        - What parts are novel vs derivative
        - Suggested structural or thematic improvements
        - Key overlooked angles
        - Research impact potential
        """
        meta_summary = _llm("openai/gpt-4.1-mini", sys_prompt, user_prompt, temperature=0.3)

        return {
            "title": title,
            "overall_novelty": overall_score,
            "section_results": results,
            "meta_summary": meta_summary.strip(),
        }

    finally:
        db.close()
