from indexing.vector_store import retrieve_documents
from core.config import TOP_K_DOCUMENTS

def get_relevant_documents(query: str):
    """Retrieve full documents from local vector store."""
    return retrieve_documents(query, top_k=TOP_K_DOCUMENTS)