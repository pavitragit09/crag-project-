from pathlib import Path

import pypdf
from docx import Document

def load_document(file_path: str) -> str:
    """Extract text from PDF, DOCX, or TXT."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    elif suffix == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")
