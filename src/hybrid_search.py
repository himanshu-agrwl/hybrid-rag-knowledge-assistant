from src.embeddings import semantic_search

def hybrid_search(
    query:str,
    chunks:list,
    model,
    chunk_embeddings,
    bm25_retriever,
    top_k: int= 2,
    candidate_k: int = 8,
    rrf_k: int = 60,

) -> list[dict]:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    semantic_results = semantic_search(
        query=query,
        chunks=chunks,
        model=model,
        chunk_embedding=chunk_embeddings,
        top_k=candidate_k,
        min_similarity=-1.0,
    )

    bm25_results = bm25_retriever.search(
        query= query,
        top_k = candidate_k
    )

    fused = {}

    for method_name , results in [
        ("semantic", semantic_results),
        ("bm25", bm25_results),
    ]:
        for rank , result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "source": result["source"],
                    "content": result["content"],
                    "rrf_score": 0.0,
                    "retrieved_by": [],
                    "semantic_score": None,
                    "bm25_score": None,
                }

            fused[chunk_id]["rrf_score"] += 1/ (rrf_k + rank)

            if method_name not in fused[chunk_id]["retrieved_by"]:
                fused[chunk_id]["retrieved_by"].append(method_name)

            fused[chunk_id][f"{method_name}_score"] = result["score"]


    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:top_k]