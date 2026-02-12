import chromadb
from chromadb.config import Settings
from core.config import CHROMA_PERSIST_DIR
from indexing.embeddings import embed_text, embed_batch

_client = None
_collection = None

def get_chroma_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name="documents",
            embedding_function=None  # We'll provide our own embeddings
        )
    return _collection

def add_document(doc_id: str, text: str, metadata: dict = None):
    """Split document into sentence chunks and add to vector store."""
    from processing.stripper import split_sentences
    sentences = split_sentences(text)
    if not sentences:
        return
    
    embeddings = embed_batch(sentences)
    metadatas = [{"doc_id": doc_id, "source": metadata.get("source", ""), "sentence": s} for s in sentences]
    
    collection = get_chroma_collection()
    collection.add(
        documents=sentences,
        embeddings=embeddings,
        ids=[f"{doc_id}_{i}" for i in range(len(sentences))],
        metadatas=metadatas
    )

def retrieve_documents(query: str, top_k: int = 5):
    """Retrieve top‑k full documents by aggregating sentence scores."""
    query_emb = embed_text(query)
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k * 5,  # over‑fetch to get full docs
        include=["documents", "metadatas", "distances"]
    )
    
    # Group by doc_id
    doc_sentences = {}
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        doc_id = meta["doc_id"]
        if doc_id not in doc_sentences:
            doc_sentences[doc_id] = {"sentences": [], "scores": [], "source": meta["source"]}
        doc_sentences[doc_id]["sentences"].append(doc)
        doc_sentences[doc_id]["scores"].append(1 - dist)  # cosine similarity
    
    # Build full document texts
    retrieved = []
    for doc_id, data in doc_sentences.items():
        full_text = " ".join(data["sentences"])
        avg_score = sum(data["scores"]) / len(data["scores"])
        retrieved.append({
            "id": doc_id,
            "text": full_text,
            "source": data["source"],
            "relevance": avg_score
        })
    
    # Sort by aggregated relevance and return top_k
    retrieved.sort(key=lambda x: x["relevance"], reverse=True)
    return retrieved[:top_k]