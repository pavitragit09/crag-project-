from groq import Groq
from core.config import GROQ_API_KEY

_client = None

def get_groq_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to the project .env file.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client
