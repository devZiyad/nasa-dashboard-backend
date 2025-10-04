import json
import numpy as np
from sklearn.cluster import KMeans
from db import SessionLocal
from models import Lesson, Publication, Section
from vector_engine import VectorEngine
from process.ai_pipeline import _llm, safe_truncate

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
        labels = lm.fit.predict(embeddings)

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
                for pub n cluster_pubs
            )
            combined_text = safe_truncate(combined_text, 8000)

            if not combined_text.strip():
                print(f"Cluster {cluster_id} has no usable text, skipping.")
                continue

            # generate lessons with llm
            system = "You are a science educator creating lessons about space biology"
            prompt = f"""
            Topic: {topic_name}

            Based on this research content, create 3 educational lessons:
            1. Beginner - Simple explanation for new learners
            2. Intermediate - deeper exploration with some technical terms
            3. Advanced - detailed expert-level overview

            Text:
            {combined_text}

            Return valid JSON like:
            [
                {{"level":"beginner","title}}
            ]