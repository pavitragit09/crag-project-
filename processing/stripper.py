import nltk
import re

def split_sentences(text: str) -> list:
    """Split text into sentence strips."""
    if not text or not text.strip():
        return []
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        # Keep the app usable when the optional Punkt data is not installed.
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
