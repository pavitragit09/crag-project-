import chromadb
from chromadb.config import Settings
from core.config import CHROMA_PERSIST_DIR
from indexing.embeddings import embed_text, embed_batch
from indexing.chunker import chunk_text

_client = None
_collection = None
BATCH_SIZE = 500

def get_chroma_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name="documents",
            embedding_function=None
        )
    return _collection

def add_document(doc_id: str, text: str, metadata: dict = None):
    """Split document into chunks and add to vector store."""
    chunks = chunk_text(text)
    if not chunks:
        print(f"No chunks generated for {doc_id}")
        return
    
    # Generate embeddings
    try:
        embeddings = embed_batch(chunks)
    except Exception as e:
        print(f"Embedding failed: {e}")
        return
    
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "source": metadata.get("source", "") if metadata else "",
            "chunk_index": i,
            "text": chunk   # store for debugging
        }
        for i, chunk in enumerate(chunks)
    ]
    
    collection = get_chroma_collection()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_embeddings = embeddings[i:i+BATCH_SIZE]
        batch_chunks = chunks[i:i+BATCH_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_SIZE]
        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_metadatas
        )
    print(f"Added {len(chunks)} chunks from '{doc_id}'.")

def retrieve_chunks(query: str, top_k: int = 5):
    """Retrieve top‑k chunks from vector store."""
    query_emb = embed_text(query)
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "id": doc_id,
            "text": doc,
            "source": meta.get("source", "unknown"),
            "doc_id": meta.get("doc_id", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "relevance": 1 - dist
        })
    return chunks