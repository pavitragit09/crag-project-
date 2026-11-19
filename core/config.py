import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Keep model downloads inside the project workspace, where the current user can
# write them and they remain available when the app is restarted.
HF_HOME = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Thresholds
# Whole-chunk relevance scores are normalized to 0.0-1.0 by document_scorer.
UPPER_THRESHOLD = float(os.getenv("UPPER_THRESHOLD", 0.7))
LOWER_THRESHOLD = float(os.getenv("LOWER_THRESHOLD", 0.4))
# Sentence/strip scores remain on the model's 0-10 scale.
STRIP_THRESHOLD = float(os.getenv("STRIP_THRESHOLD", 5.0))

# Retrieval
TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", 5))
TOP_K_WEB_RESULTS = int(os.getenv("TOP_K_WEB_RESULTS", 3))

# Models
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "openai/gpt-oss-120b")

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_data"))
