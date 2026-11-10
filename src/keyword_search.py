import re

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with"
}

def tokenize(text:str) -> list[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [word for word in words if word not in STOP_WORDS]

def keyword_search(
        query:str,
        chunks:list,
        top_k: int= 3,

) -> list[dict]:

    if not query.strip():
        raise ValueError("Query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return[]

    query_terms = set(tokenize(query))

    if not query_terms:
        return[]

    results = []

    for chunk in chunks:
        chunk_terms =set(tokenize(chunk.content))
        matched_terms = query_terms.intersection(chunk_terms)

        score = len(matched_terms)/ len(query_terms)

        if score ==0:
            continue

        results.append({
            "score": score,
            "matched_terms": sorted(matched_terms),
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
        })

        results.sort(key = lambda result: result["score"], reverse = True)

        return results[:top_k]
