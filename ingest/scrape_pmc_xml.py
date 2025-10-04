import asyncio
import aiohttp
import re
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from utils.text_clean import normalize_whitespace
from models import Publication, Section, SectionType
from db import SessionLocal
from config import Config
from difflib import get_close_matches
from transformers import pipeline

HEADERS = {"User-Agent": "NASA-BioDash/1.0"}

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = 0 if getattr(Config, "DEVICE", "cpu") == "cuda" else -1
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", device=device)


def pmc_id_from_url(url: str) -> str | None:
    m = re.search(r'PMC(\d+)', url)
    return f"PMC{m.group(1)}" if m else None


SECTION_LABELS = [
    "abstract", "introduction", "methods",
    "results", "discussion", "conclusion"
]


def classify_section_ai(title: str, text: str = "") -> str:
    title = (title or "").strip()
    text = (text or "").strip()

    # --- Case 1: Use title if available ---
    if title:
        result = classifier(title, SECTION_LABELS)
        label = result["labels"][0]
        score = result["scores"][0]

        if score >= 0.6:
            return label

    # --- Case 2: No title, but text exists → use snippet ---
    if not title and text:
        snippet = text[:200]  # first 200 chars as proxy
        result = classifier(snippet, SECTION_LABELS)
        label = result["labels"][0]
        score = result["scores"][0]
        if score >= 0.5:
            return label

    # --- Case 3: Fallback simple rules ---
    low_title = title.lower()
    if "method" in low_title or "materials" in low_title:
        return "methods"
    if "result" in low_title or "finding" in low_title:
        return "results"
    if "discuss" in low_title or "analysis" in low_title:
        return "discussion"
    if "intro" in low_title or "background" in low_title:
        return "introduction"
    if "conclusion" in low_title or "summary" in low_title:
        return "conclusion"

    return "other"


async def fetch_xml(session: aiohttp.ClientSession, pmc_id: str) -> str | None:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={
        pmc_id}"
    if Config.NCBI_API_KEY:
        url += f"&api_key={Config.NCBI_API_KEY}"

    try:
        async with asyncio.timeout(30):
            async with session.get(url, headers=HEADERS) as resp:
                if resp.status == 429:
                    logger.warning(f"Rate limited when fetching {
                                   pmc_id}, sleeping 0.5s...")
                    await asyncio.sleep(0.5)
                    return await fetch_xml(session, pmc_id)
                if resp.status != 200:
                    logger.error(f"Failed fetch {
                                 pmc_id} (status={resp.status})")
                    return None
                text = await resp.text()
                if not text.strip():
                    logger.error(f"Empty response for {pmc_id}")
                    return None
                return text
    except Exception as e:
        logger.exception(f"Exception fetching {pmc_id}: {e}")
        return None


def parse_xml(xml_str: str) -> Dict[str, Any]:
    soup = BeautifulSoup(xml_str, "lxml-xml")
    article = soup.find("article")
    if not article:
        raise ValueError("No <article> root found in XML")

    # detect restriction: no <body>
    restricted = article.find("body") is None

    # ---- Title ----
    title_tag = article.find("article-title")
    title = normalize_whitespace(title_tag.get_text(
        " ", strip=True)) if title_tag else ""

    # ---- Journal ----
    journal_tag = article.find("journal-title")
    journal = normalize_whitespace(journal_tag.get_text(
        " ", strip=True)) if journal_tag else None

    # ---- Publication Date ----
    def extract_date(tag):
        if not tag:
            return None
        y = tag.find("year")
        m = tag.find("month")
        d = tag.find("day")
        return {
            "year": int(y.text) if y and y.text.isdigit() else None,
            "month": int(m.text) if m and m.text.isdigit() else None,
            "day": int(d.text) if d and d.text.isdigit() else None,
        }

    epub_date = article.find("pub-date", {"pub-type": "epub"})
    collection_date = article.find("pub-date", {"pub-type": "collection"})
    other_date = article.find("pub-date")

    pub_date = extract_date(epub_date) or extract_date(
        collection_date) or extract_date(other_date)
    year = pub_date["year"] if pub_date else None

    # ---- License ----
    license_tag = article.find("license")
    license_txt = normalize_whitespace(license_tag.get_text(
        " ", strip=True)) if license_tag else None

    # ---- Sections ----
    sections = []
    for abstract_tag in article.find_all("abstract"):
        abstract_txt = normalize_whitespace(
            abstract_tag.get_text(" ", strip=True))
        if abstract_txt:
            sections.append({"kind": "abstract", "text": abstract_txt})

    for sec in article.find_all("sec"):
        head = sec.find("title")
        head_txt = head.get_text(" ", strip=True).lower() if head else ""
        txt = normalize_whitespace(sec.get_text(" ", strip=True))
        kind = classify_section_ai(head_txt, txt)
        sections.append({"kind": kind, "text": txt})


async def crawl_and_store(urls: List[str]) -> Dict[str, Any]:
    out = {"ok": 0, "fail": 0}
    connector = aiohttp.TCPConnector(limit=6)
    async with aiohttp.ClientSession(connector=connector) as session:
        db: Session = SessionLocal()
        try:
            for url in urls:
                pmc_id = pmc_id_from_url(url)
                if not pmc_id:
                    logger.warning(f"Could not extract PMC ID from {url}")
                    out["fail"] += 1
                    continue

                logger.info(f"Fetching {pmc_id} from XML API...")
                xml_str = await fetch_xml(session, pmc_id)
                if not xml_str:
                    out["fail"] += 1
                    continue

                parsed = parse_xml(xml_str)
                if not parsed:
                    logger.error(f"Parsing failed for {pmc_id}")
                    out["fail"] += 1
                    continue

                pub = db.query(Publication).filter(
                    Publication.pmc_id == pmc_id).one_or_none()
                if not pub:
                    pub = Publication(
                        pmc_id=pmc_id,
                        title=parsed["title"],
                        link=url,
                        journal=parsed["journal"],
                        year=parsed["year"],
                        xml_restricted=parsed["xml_restricted"]
                    )
                    db.add(pub)
                    db.flush()
                else:
                    pub.title = parsed["title"] or pub.title
                    pub.journal = parsed["journal"] or pub.journal
                    pub.year = parsed["year"] or pub.year
                    pub.xml_restricted = parsed["xml_restricted"]

                # replace sections
                db.query(Section).filter(
                    Section.publication_id == pub.id).delete()
                for s in parsed["sections"]:
                    try:
                        st = SectionType(s["kind"])
                    except ValueError:
                        st = SectionType.other
                    db.add(Section(publication_id=pub.id,
                           kind=st, text=s["text"]))

                out["ok"] += 1

            db.commit()
        finally:
            db.close()
    return out
