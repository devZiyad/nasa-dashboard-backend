from typing import Dict, List

MODES = {"conservative": 0.35, "balanced": 0.22, "aggressive": 0.10}

def simulate(question: str, context: List[Dict], knobs: Dict) -> Dict:
    k = int(knobs.get("evidence_k", 5))
    mode = knobs.get("mode", "balanced")
    require_kw = str(knobs.get("require_keyword", "")).strip().lower()
    prefer = [s.lower() for s in knobs.get("prefer_kinds", ["results","discussion"])]
    threshold = MODES.get(mode, 0.22)

    items = []
    for r in context:
        snip = r.get("snippet","")
        score = r.get("score", 0.0)
        if score < threshold:
            continue
        if require_kw and require_kw not in snip.lower():
            continue
        bonus = sum(0.05 for pk in prefer if f"[{pk.upper()}]" in snip)
        items.append((score + bonus, r))

    items.sort(key=lambda x: x[0], reverse=True)
    evidence = [r for _, r in items[:k]]

    if not evidence:
        return {
            "answer": "No strong evidence found.",
            "confidence": 0.2,
            "rationale": [],
            "controls": {"applied": knobs},
        }

    cues = []
    seen_titles = set()
    for e in evidence:
        title = e.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        frag = e.get("snippet","")[:240].split("\n", 1)[-1]
        cues.append(f"• {frag} (→ {title}, {e.get('year','?')})")

    return {
        "answer": "Based on the strongest retrieved results:\n" + "\n".join(cues[:6]),
        "confidence": min(0.95, max(ev.get("score",0) for ev in evidence)),
        "rationale": evidence,
        "controls": {"applied": knobs},
    }
