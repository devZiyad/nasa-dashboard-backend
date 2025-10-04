import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session, joinedload
from models import Publication, Section
from db import SessionLocal
from config import Config


class VectorEngine:
    def __init__(self, persist: bool = True):
        self.persist = persist
        # fmt: off
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=Config.DEVICE)

        # 🔹 warm-up encoding so first query is not slow
        _ = self.model.encode(["warmup"], convert_to_numpy=True)

        self.index = None
        self.id_map = None
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
            # preload publications to avoid n+1 lookups
            sections = (
                db.query(Section)
                .options(joinedload(Section.publication))
                .all()
            )

            texts, ids = [], []

            for s in sections:
                if not s.text or not s.text.strip():
                    continue
                texts.append(s.text.strip())
                pub = s.publication
                ids.append({
                    "pub_id": s.publication_id,
                    "section": s.kind.value,
                    "journal": pub.journal if pub else None,
                    "year": pub.year if pub else None,
                    "restricted": pub.xml_restricted if pub else None,
                })

            if not texts:
                self.embeddings = np.empty((0, 384), dtype="float32")
                self.id_map = np.empty((0,), dtype=object)
                self.index = faiss.IndexFlatL2(384)
                return

            self.embeddings = self.model.encode(
                texts, convert_to_numpy=True).astype("float32")
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
    def search(
        self,
        query: str,
        top_k: int = 10,
        section: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        journal: str | None = None,
        restricted: bool | None = None,
    ):
        if self.index is None or self.id_map is None:
            return []

        # fmt: off
        qvec = self.model.encode([query], convert_to_numpy=True).astype("float32")

        # Fetch more candidates than needed in case filters drop some
        D, I = self.index.search(qvec, top_k * 5)
        results = []

        for dist, idx in zip(D[0], I[0]):
            meta = self.id_map[idx]

            # Apply filters
            if section and meta["section"] != section:
                continue
            if year_from and (not meta["year"] or meta["year"] < year_from):
                continue
            if year_to and (not meta["year"] or meta["year"] > year_to):
                continue
            if journal and (not meta["journal"] or journal.lower() not in meta["journal"].lower()):
                continue
            if restricted is not None and meta["restricted"] != restricted:
                continue

            results.append((meta["pub_id"], meta["section"], float(dist)))
            if len(results) >= top_k:
                break

        return results
