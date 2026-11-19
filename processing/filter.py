from core.groq_client import get_groq_client
from core.config import GENERATOR_MODEL, STRIP_THRESHOLD
from utils.prompts import FILTER_PROMPT
from utils.helpers import parse_relevance_score

def score_strip(query: str, strip: str) -> float:
    """Use LLM to rate relevance of a single strip (0‑10)."""
    client = get_groq_client()
    prompt = FILTER_PROMPT.format(query=query, strip=strip)
    
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        extra_body={"reasoning_effort": "low", "include_reasoning": False},
    )
    return parse_relevance_score(response.choices[0].message.content)

def filter_strips(query: str, strips: list) -> list:
    """Keep only strips with score ≥ STRIP_THRESHOLD."""
    kept = []
    for strip in strips:
        score = score_strip(query, strip)
        if score >= STRIP_THRESHOLD:
            kept.append((strip, score))
    return kept
