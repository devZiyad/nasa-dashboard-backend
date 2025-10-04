from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Enum, Boolean, Float
import enum
from db import engine


class Base(DeclarativeBase):
    pass


class SectionType(enum.Enum):
    abstract = "abstract"
    introduction = "introduction"
    methods = "methods"
    results = "results"
    discussion = "discussion"
    conclusion = "conclusion"
    other = "other"


class Entity(Base):
    __tablename__ = "entity"

    # fmt: off
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # fmt: off
    publication_id: Mapped[int] = mapped_column(ForeignKey("publication.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # fmt: off
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g. "organism", "outcome"

    publication = relationship("Publication", back_populates="entities")

class Triple(Base):
    __tablename__ = "triple"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    publication_id: Mapped[int] = mapped_column(ForeignKey("publication.id"), nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    relation: Mapped[str] = mapped_column(Text, nullable=False)
    object: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    publication = relationship("Publication", back_populates="triples")

class Lesson(Base):
    __tablename__ = "lesson"

    id = Column(Integer, primary_key = True)
    topic = Column(String)
    title = Column(Text)
    level = Column(String)
    content = Column(Text)
    difficulty_score = Column(Float, nullable=True)
    publication_ids = Column(Text)

    questions = relationship("Question", back_populates="lesson",cascade="all, delete-orphan")

class Question(Base):
    __tablename__ = "question"

    id = Column(Integer, primary_key=True)
    lesson_id = Column(Integer, ForeignKey("lesson.id"))
    text = Column(Text)
    choices = Column(Text)
    answer = Column(Text)
    difficulty = Column(String)

    lesson = relationship("Lesson", back_populates="questions")

class UserProgress(Base):
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True)
    username = Column(String)
    lesson_id = Column(Integer, ForeignKey("lesson.id"))
    score = Column(Float)
    completed = Column(Boolean, default=False)
    badge = Column(String, nullable=True)


class Publication(Base):
    __tablename__ = "publication"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True)
    pmc_id: Mapped[str | None] = mapped_column(String, unique=True)
    title: Mapped[str] = mapped_column(Text)
    link: Mapped[str | None] = mapped_column(Text)
    journal: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    license: Mapped[str | None] = mapped_column(Text)
    raw_html_uri: Mapped[str | None] = mapped_column(Text)
    raw_pdf_uri: Mapped[str | None] = mapped_column(Text)

    xml_restricted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False)

    sections: Mapped[list["Section"]] = relationship(
        back_populates="publication", cascade="all, delete-orphan"
    )

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    entities: Mapped[list["Entity"]] = relationship(
        back_populates="publication", cascade="all, delete-orphan"
    )

    triples: Mapped[list["Triple"]] = relationship(
        back_populates="publication", cascade="all, delete-orphan"
    )


class Section(Base):
    __tablename__ = "section"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    publication_id: Mapped[int] = mapped_column(ForeignKey("publication.id"))
    kind: Mapped[SectionType] = mapped_column(Enum(SectionType))
    text: Mapped[str] = mapped_column(Text)

    publication: Mapped[Publication] = relationship(back_populates="sections")


def init_db():
    Base.metadata.create_all(engine)
