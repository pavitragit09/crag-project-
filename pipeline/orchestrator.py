from retrieval.retriever import get_relevant_documents
from retrieval.web_search import web_search
from evaluation.document_scorer import score_document_relevance
from evaluation.decision import classify_documents
from processing.stripper import split_sentences
from processing.filter import filter_strips, score_strip
from processing.merger import merge_kept_strips
from generation.generator import generate_answer
from utils.prompts import REWRITE_PROMPT
from core.groq_client import get_groq_client
from core.config import LOWER_THRESHOLD, UPPER_THRESHOLD, EVALUATOR_MODEL

def rewrite_query(query: str) -> str:
    """Expand query for web search."""
    client = get_groq_client()
    prompt = REWRITE_PROMPT.format(query=query)
    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def refine_document(doc: dict, query: str) -> dict:
    """Split, filter, merge a single document."""
    strips = split_sentences(doc["text"])
    kept = filter_strips(query, strips)
    if not kept:
        return None
    return merge_kept_strips(kept, doc.get("source", doc.get("url", "unknown")))

def answer_query(query: str):
    """Main pipeline: retrieve → evaluate → classify → refine → generate."""
    # 1. Retrieve local documents
    local_docs = get_relevant_documents(query)
    
    # 2. Score each document (whole‑doc evaluation)
    doc_scores = []
    for doc in local_docs:
        score = score_document_relevance(query, doc["text"])
        doc_scores.append((doc, score))
    
    # 3. Classify overall result
    classification = classify_documents(doc_scores)
    
    context_pieces = []
    sources = []
    
    # 4. Conditional refinement & web fallback
    if classification == "correct":
        # Use only documents with score ≥ UT
        best_docs = [doc for doc, score in doc_scores if score >= UPPER_THRESHOLD]
        for doc in best_docs[:1]:  # take top 1 for simplicity
            refined = refine_document(doc, query)
            if refined:
                context_pieces.append(refined["text"])
                sources.append({
                    "source": doc.get("source", "local"),
                    "strips_used": refined["strips_used"]
                })
    
    elif classification == "ambiguous":
        # Refine all local docs with score > LT
        useful_local = [doc for doc, score in doc_scores if score > LOWER_THRESHOLD]
        for doc in useful_local:
            refined = refine_document(doc, query)
            if refined:
                context_pieces.append(refined["text"])
                sources.append({
                    "source": doc.get("source", "local"),
                    "strips_used": refined["strips_used"]
                })
        
        # Web search
        rewritten = rewrite_query(query)
        web_docs = web_search(rewritten)
        for wdoc in web_docs:
            refined = refine_document(wdoc, query)
            if refined:
                context_pieces.append(refined["text"])
                sources.append({
                    "source": wdoc["source"],
                    "title": wdoc.get("title", ""),
                    "url": wdoc["source"],
                    "strips_used": refined["strips_used"]
                })
    
    else:  # incorrect
        # No local refinement, only web
        rewritten = rewrite_query(query)
        web_docs = web_search(rewritten)
        for wdoc in web_docs:
            refined = refine_document(wdoc, query)
            if refined:
                context_pieces.append(refined["text"])
                sources.append({
                    "source": wdoc["source"],
                    "title": wdoc.get("title", ""),
                    "url": wdoc["source"],
                    "strips_used": refined["strips_used"]
                })
    
    # 5. Generation (streaming)
    final_context = "\n\n---\n\n".join(context_pieces) if context_pieces else "No relevant information found."
    
    # Return both the streaming generator and the sources
    return generate_answer(query, final_context), sources