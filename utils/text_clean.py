import re


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def safe_truncate(s: str, n: int = 4000) -> str:
    if s and len(s) > n:
        return s[:n] + "..."
    return s
