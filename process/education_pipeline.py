import json
import numpy as np
from sklearn.cluster import KMeans
from db import SessionLocal
from models import Lesson, Publication, Section
from vector_engine import VectorEngine
from process.ai_pipeline import _llm, safe_truncate

def safe_parse_json(raw_text: str):
    """
    Safely parse JSON returned from LLM responses that may include ```json fences.
    Returns a Python object or None.
    """
    if not raw_text:
        return None

    raw = raw_text.strip()

    # Handle markdown code fences like ```json ... ```
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]  # strip the first fenced block content
        raw = raw.replace("json", "", 1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print("Raw (truncated):", raw[:400])
        return None

def generate_lessons_for_all_topics(n_clusters: int = 10):
    ## clusters publications based on title and summary then gens 3 lessons in 3 diffs for each topic using full text

    db = SessionLocal()
    lessons_created= []

    try:
        pubs = db.query(Publication).filter(Publication.summary.isnot(None)).all()
        if not pubs:
            print("No publications with summaries found, please create summaries first")
            return []

        print(f" LOaded {len(pubs)} publications with sumamries")

        # build text for clustering in form of title+summary
        text_corpus = [f"{p.title}.{p.summary}" for p in pubs]

        #generate embeds
        ve = VectorEngine(persist=True)
        embeddings = ve.model.encode(text_corpus, convert_to_numpy=True)
       
        # Kmeans clustering
        n_clusters = min(n_clusters, len(embeddings))
        km = KMeans(n_clusters=n_clusters, random_state = 42)
        labels = km.fit_predict(embeddings)

        #group publications by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(pubs[idx])

        print(f" Created {len(clusters)} topic clusters")

        # generate lessons for each cluster

        for cluster_id, cluster_pubs in clusters.items():
            topic_name = generate_topic_name(cluster_pubs)
            print(f" Generating Lessons for Cluster {cluster_id}: {topic_name}")

            combined_text = " ".join(
                " ".join(s.text for s in pub.sections if s.text)
                for pub in cluster_pubs
            )
            combined_text = safe_truncate(combined_text, 8000)

            if not combined_text.strip():
                print(f"Cluster {cluster_id} has no usable text, skipping.")
                continue

            # generate lessons with llm
            system = "You are a science educator creating lessons about space biology."
            prompt = f"""
            Topic: {topic_name}

            Based on this research content, create 3 educational lessons:
            1. Beginner — simple explanation for new learners
            2. Intermediate — deeper exploration with some technical terms
            3. Advanced — detailed expert-level overview

            Text:
            {combined_text}

            Return valid JSON like:
            [
                {{"level": "beginner", "title": "...", "content": "..." }},
                {{"level": "intermediate", "title": "...", "content": "..." }},
                {{"level": "advanced", "title": "...", "content": "..." }}
            ]
            """
            
            raw_json = _llm("openai/gpt-4o-mini", system, prompt)
            
            if not raw_json or raw_json.strip() == "" or "LLM error" in raw_json:
                print(f"Skipping cluster {cluster_id} — empty or invalid LLM response")
                continue
            
            try:
               # lessons = json.loads(raw_json)
               lessons = safe_parse_json(raw_json)
               if not lessons:
                   print(f"Skipping cluster {cluster_id} — invalid or empty JSON")
                   continue
                
            except Exception as e:
                print(f"Failed to parse JSON for cluster {cluster_id}: {e}")
                print(f"Raw content returned by LLM:\n{raw_json[:500]}...\n")
                lessons = []
                continue
            
            #try:
             #   lessons=json.loads(raw_json)
            #except Exception:
             #   print(f" Failed to parse JSON for cluster {cluster_id}: {e}")
             #   continue
            
            # save lessons to db
            
            for lesson_data in lessons:
                new_lesson = Lesson(
                    topic = topic_name,
                    title = lesson_data.get("title"),
                    content = lesson_data.get("content"),
                    level = lesson_data.get("level"),
                    publication_ids =",".join(str(p.id) for p in cluster_pubs)
                )
                db.add(new_lesson)
                db.commit()
                db.refresh(new_lesson)
                lessons_created.append(new_lesson)
                print(f"  Created lesson: {new_lesson.title} ({new_lesson.level})")
        print(f" Created total {len(lessons_created)} lessons across {len(clusters)} topics")
        db.commit()
        
    finally:
        db.close()
        
    return lessons_created
        
def generate_topic_name(publications):

    titles = " ".join(p.title for p in publications if p.title)
    titles = safe_truncate(titles, 2000)

    system = "You are a helpful assistant that generates concise topic names."
    prompt = f"""
    Given these publication titles, generate a concise topic name (3-5 words) that captures their common theme:

    Titles:
    {titles}

    Topic Name:
    """
    topic_name = _llm("openai/gpt-4o-mini", system, prompt)
    return topic_name.strip().strip('"')