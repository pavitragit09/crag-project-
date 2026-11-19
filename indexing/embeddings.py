import numpy as np
from pathlib import Path
from core.config import EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer

_model = None

_local_model = Path(__file__).resolve().parents[1] / ".cache" / "models" / "all-MiniLM-L6-v2"
_model_source = str(_local_model) if (_local_model / "modules.json").is_file() else EMBEDDING_MODEL

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_model_source)
    return _model

def embed_text(text: str) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(text).tolist()

def embed_batch(texts: list) -> list:
    model = get_embedding_model()
    return model.encode(texts).tolist()
