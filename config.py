import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    PORT = int(os.getenv("PORT", "5000"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./biodash.db")
    DEVICE = os.getenv("DEVICE", "cpu")

    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./biodash_faiss.index")
    EMBEDDINGS_NPY_PATH = os.getenv(
        "EMBEDDINGS_NPY_PATH", "./biodash_embeddings.npy")
    ID_MAP_NPY_PATH = os.getenv("ID_MAP_NPY_PATH", "./biodash_idmap.npy")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    # NCBI API key
    NCBI_API_KEY = os.getenv("NCBI_API_KEY", None)
