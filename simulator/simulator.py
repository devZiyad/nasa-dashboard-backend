from typing import Dict, List
from collections import Counter

# existing + two extra modes (still default to 'balanced')
MODES = {
    "conservative": 0.35,   # high precision
    "balanced": 0.22,       # default
    "aggressive": 0.10,     # high recall
    "strict": 0.30,         # stricter than balanced but not too harsh
    "exploratory": 0.12,    # looser than aggressive
}

def _tally_entities(context: List[Dict]) -> Dict[str, List[Dict]]:
    """Make a compact, UI-friendly entity summary from retrieved items."""
    buckets = {}
    for r in context:
        ents = r.get("entities", {})
        for etype, values in ents.items():
            buckets.setdefault(etype, Counter()).update([v.lower() for v in values][:20])
    ranked = {}
    for etype, counter in buckets.items():
        top = counter.most_common(8)
        ranked[etype] = [{"value": v, "count": int(c)} for v, c in top]
    return ranked

def simulate(question: str, context: List[Dict], knobs: Dict) -> Dict:
    k = int(knobs.get("evidence_k", 5))
    mode = knobs.get("mode", "balanced")
    require_kw = str(knobs.get("require_keyword", "")).strip().lower()
    prefer = [s.lower() for s in knobs.get("prefer_kinds", ["results", "discussion"])]
    threshold = MODES.get(mode, MODES["balanced"])

    items = []
    for r in context:
        snip = r.get("snippet", "")
        score = r.get("score", 0.0)
        if score < threshold:
            continue
        if require_kw and require_kw not in snip.lower():
            continue
        # small bonus for preferred section tags
        bonus = sum(0.05 for pk in prefer if f"[{pk.upper()}]" in snip)
        items.append((score + bonus, r))

    items.sort(key=lambda x: x[0], reverse=True)
    evidence = [r for _, r in items[:k]]

    if not evidence:
        return {
            "answer": "No strong evidence found.",
            "confidence": 0.2,
            "rationale": [],
            "controls": {"applied": {"evidence_k": k, "mode": mode, "require_keyword": require_kw}},
            "entities": {},  # reasoning panel stays empty if no evidence
        }

    # Build the human-readable answer bullets and reasoning payload
    cues, seen_titles = [], set()
    for e in evidence:
        title = e.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        frag = e.get("snippet", "")[:240].split("\n", 1)[-1]
        cues.append(f"• {frag} (→ {title}, {e.get('year', '?')})")

    answer = "Based on the strongest retrieved results:\n" + "\n".join(cues[:6])
    conf = min(0.95, max(ev.get("score", 0) for ev in evidence))
    entity_panel = _tally_entities(evidence)

    # enrich rationale entries a bit
    enriched = []
    for e in evidence:
        kind = "OTHER"
        sn = e.get("snippet", "")
        if "[RESULTS]" in sn:
            kind = "RESULTS"
        elif "[DISCUSSION]" in sn:
            kind = "DISCUSSION"
        enriched.append({
            "publication_id": e.get("publication_id"),
            "title": e.get("title"),
            "year": e.get("year"),
            "journal": e.get("journal"),
            "link": e.get("link"),
            "kind": kind,
            "score": round(float(e.get("score", 0)), 4),
            "snippet": sn,
        })

    return {
        "answer": answer,
        "confidence": round(float(conf), 3),
        "rationale": enriched,
        "controls": {"applied": {"evidence_k": k, "mode": mode, "require_keyword": require_kw}},
        "entities": entity_panel,   # ← reasoning panel: entity tallies
    }
