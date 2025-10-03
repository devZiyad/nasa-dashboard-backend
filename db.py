from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import Config

engine = create_engine(Config.DATABASE_URL, future=True, echo=False)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True)
