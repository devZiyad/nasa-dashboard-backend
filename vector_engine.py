import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from models import Publication, Section
from db import SessionLocal
from config import Config


class VectorEngine:
    def __init__(self, persist: bool = True):
        self.persist = persist
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        # numpy array mapping faiss_id -> (pub_id, section_id, kind)
        self.meta = None
        if persist and self._load_index():
            return
        self._build_index()

    def _load_index(self) -> bool:
        if not (os.path.exists(Config.FAISS_INDEX_PATH)
                and os.path.exists(Config.EMBEDDINGS_NPY_PATH)
                and os.path.exists(Config.ID_MAP_NPY_PATH)):
            return False
        self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
        self.embeddings = np.load(Config.EMBEDDINGS_NPY_PATH)
        self.meta = np.load(Config.ID_MAP_NPY_PATH, allow_pickle=True)
        return True

    def _build_index(self):
        db: Session = SessionLocal()
        try:
            texts, meta = [], []
            pubs = db.query(Publication).all()

            for p in pubs:
                sections = db.query(Section).filter(
                    Section.publication_id == p.id).all()
                for s in sections:
                    if not s.text or not s.text.strip():
                        continue
                    texts.append(s.text)
                    meta.append((p.id, s.id, s.kind.value))

            if not texts:
                self.embeddings = np.empty((0, 384), dtype="float32")
                self.meta = np.empty((0,), dtype=object)
                self.index = faiss.IndexFlatL2(384)
                return

            print(f"⚡ Embedding {len(texts)} sections...")
            self.embeddings = self.model.encode(
                texts, convert_to_numpy=True).astype("float32")
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            self.meta = np.array(meta, dtype=object)

            if self.persist:
                faiss.write_index(self.index, Config.FAISS_INDEX_PATH)
                np.save(Config.EMBEDDINGS_NPY_PATH, self.embeddings)
                np.save(Config.ID_MAP_NPY_PATH, self.meta)
        finally:
            db.close()

    def search(self, query: str, top_k: int = 10):
        if self.index is None or self.meta is None:
            return []
        qvec = self.model.encode(
            [query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(qvec, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            pub_id, section_id, kind = self.meta[idx]
            results.append({
                "pub_id": int(pub_id),
                "section_id": int(section_id),
                "kind": kind,
                "distance": float(dist)
            })
        return results
