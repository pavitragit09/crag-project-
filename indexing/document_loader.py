import pypdf
from docx import Document

def load_document(file_path: str) -> str:
    """Extract text from PDF, DOCX, or TXT."""
    if file_path.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:  # assume .txt
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()