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
        self.id_map = None   # [(pub_id, section_kind), ...]
        self.embeddings = None
        if persist and self._load_index():
            return
        self._build_index()

    # ----------------------
    # Persistence
    # ----------------------
    def _load_index(self) -> bool:
        if not (
            os.path.exists(Config.FAISS_INDEX_PATH)
            and os.path.exists(Config.EMBEDDINGS_NPY_PATH)
            and os.path.exists(Config.ID_MAP_NPY_PATH)
        ):
            return False
        self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
        self.embeddings = np.load(
            Config.EMBEDDINGS_NPY_PATH, allow_pickle=True)
        self.id_map = np.load(Config.ID_MAP_NPY_PATH, allow_pickle=True)
        return True

    def _save_index(self):
        if not self.persist:
            return
        faiss.write_index(self.index, Config.FAISS_INDEX_PATH)
        np.save(Config.EMBEDDINGS_NPY_PATH, self.embeddings)
        np.save(Config.ID_MAP_NPY_PATH, self.id_map)

    # ----------------------
    # Index Building
    # ----------------------
    def _build_index(self):
        db: Session = SessionLocal()
        try:
            sections = db.query(Section).all()
            texts, ids = [], []

            for s in sections:
                if not s.text or not s.text.strip():
                    continue
                texts.append(s.text.strip())
                ids.append((s.publication_id, s.kind.value))

            if not texts:
                self.embeddings = np.empty((0, 384), dtype="float32")
                self.id_map = np.empty((0,), dtype=object)
                self.index = faiss.IndexFlatL2(384)
                return

            self.embeddings = self.model.encode(
                texts, convert_to_numpy=True
            ).astype("float32")
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            self.id_map = np.array(ids, dtype=object)

            self._save_index()
        finally:
            db.close()

    # ----------------------
    # Search
    # ----------------------
    def search(self, query: str, top_k: int = 10, section: str | None = None):
        if self.index is None or self.id_map is None:
            return []

        qvec = self.model.encode(
            [query], convert_to_numpy=True).astype("float32")

        # If section is specified → filter embeddings before searching
        if section:
            mask = [i for i, (_, kind) in enumerate(
                self.id_map) if kind == section]
            if not mask:
                return []

            sub_embeds = self.embeddings[mask]
            sub_index = faiss.IndexFlatL2(sub_embeds.shape[1])
            sub_index.add(sub_embeds)

            D, I = sub_index.search(qvec, top_k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                pub_id, kind = self.id_map[mask[idx]]
                results.append((int(pub_id), kind, float(dist)))
            return results

        # Global search
        D, I = self.index.search(qvec, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            pub_id, kind = self.id_map[idx]
            results.append((int(pub_id), kind, float(dist)))
        return results
