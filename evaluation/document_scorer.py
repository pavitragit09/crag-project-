from core.groq_client import get_groq_client
from core.config import GENERATOR_MODEL
from utils.prompts import SCORE_DOC_PROMPT
from utils.helpers import parse_relevance_score

def score_document_relevance(query: str, document_text: str) -> float:
    """Return whole-chunk relevance normalized to 0.0-1.0.

    The scoring prompt asks the LLM for a 0-10 score; normalize it so the
    routing thresholds can be expressed as 0.7 and 0.4.
    """
    client = get_groq_client()
    prompt = SCORE_DOC_PROMPT.format(query=query, document=document_text)
    
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        extra_body={"reasoning_effort": "low", "include_reasoning": False},
    )
    return parse_relevance_score(response.choices[0].message.content) / 10.0
