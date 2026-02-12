FILTER_PROMPT = """You are a strict relevance judge. Rate the relevance of the following sentence to the query on a scale of 0 to 10.
Only output a single number, nothing else.

Query: {query}
Sentence: {strip}
Relevance score (0‑10):"""

SCORE_DOC_PROMPT = """You are a strict relevance judge. Rate the relevance of the following document to the query on a scale of 0 to 10.
Only output a single number, nothing else.

Query: {query}
Document:
{document}
Relevance score (0‑10):"""

REWRITE_PROMPT = """Rewrite the user's query to be more effective for a web search engine. Output only the rewritten query, no explanations.
Original query: {query}
Rewritten query:"""

GENERATE_PROMPT = """Answer the question based **only** on the provided context. If the context does not contain the answer, say "I don't have enough information to answer this." At the end, cite the exact sentences from the context that support your answer.

Context:
{context}

Question: {query}

Answer:"""