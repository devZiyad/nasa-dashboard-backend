import os
import requests
from config import Config
from utils.text_clean import safe_truncate


def _llm(model: str, system: str, user: str, temperature: float = 0.2) -> str:
    if not Config.OPENROUTER_API_KEY:
        return "[OpenRouter key missing] Skipping."

    headers = {
        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }

    try:
        r = requests.post(
            Config.OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM error: {e}]"


def summarize_paper(title: str, abstract: str, results_text: str) -> str:
    system = "Summarize the study in 4–6 bullet points focusing on objective results. Neutral, precise language."
    prompt = (
        f"Title: {title}\n\n"
        f"Abstract:\n{safe_truncate(abstract)}\n\n"
        f"Results:\n{safe_truncate(results_text)}"
    )
    return _llm("openai/gpt-4.1-mini", system, prompt)


def extract_entities_triples(text: str) -> str:
    system = (
        "Extract entities (organism, tissue/system, intervention, outcome, measurement) "
        "and relation triples as JSON. Always include both 'entities' and 'triples'. "
        "If no relation is found, infer possible relations from context. "
        "Output strictly valid JSON like: "
        "{\"entities\": [{\"text\":...,\"type\":...}], "
        "\"triples\": [{\"subject\":...,\"relation\":...,\"object\":...,"
        "\"evidence_sentence\":...,\"confidence\":0.0}]}"
    )
    return _llm("openai/gpt-4.1-mini", system, safe_truncate(text, 8000))


def chat_with_context(messages: list[dict], model: str = "openai/gpt-4.1-mini", temperature: float = 0.2) -> str:
    """Generic chat function using the same OpenRouter backend."""
    if not Config.OPENROUTER_API_KEY:
        return "[OpenRouter key missing] Skipping."

    headers = {
        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        r = requests.post(
            Config.OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM error: {e}]"
