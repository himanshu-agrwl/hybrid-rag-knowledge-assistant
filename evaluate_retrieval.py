
import json
from pathlib import Path

from src.document_loader import load_documents
from src.chunker import paragraph_chunks
from src.embeddings import load_embedding_model, embed_text, semantic_search
from src.bm25_search import BM25Retriever
from src.hybrid_search import hybrid_search


def recall_at_k(results: list[dict], relevant_ids: set[str], k: int = 3) -> float:
    """Calculate the fraction of known relevant chunks found in top-k."""
    if not relevant_ids:
        raise ValueError("Each query must have at least one relevant chunk ID")

    retrieved_ids = {result["chunk_id"] for result in results[:k]}
    return len(retrieved_ids & relevant_ids) / len(relevant_ids)



def precision_at_k(
    results: list[dict],
    relevant_ids: set[str],
    k: int = 3,
) -> float:
    """Fraction of the top-k positions occupied by relevant chunks."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    retrieved_ids = {result["chunk_id"] for result in results[:k]}
    relevant_retrieved = len(retrieved_ids & relevant_ids)

    # Standard Precision@k uses k as the denominator,
    # even when fewer than k results are returned.
    return relevant_retrieved / k


def reciprocal_rank(
    results: list[dict],
    relevant_ids: set[str],
) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in relevant_ids:
            return 1.0 / rank

    return 0.0



def main():
    # 1. Load the evaluation questions.
    evaluation_path = Path("data/evaluation_queries.json")
    with evaluation_path.open("r", encoding="utf-8") as file:
        evaluation_queries = json.load(file)

    # 2. Load documents and recreate the same chunks as main.py.
    documents = load_documents("data/documents")

    chunks = []
    for document in documents:
        chunks.extend(paragraph_chunks(document, max_chars=300))

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    # 3. Build the retrieval components once.
    model = load_embedding_model()
    chunk_embeddings = embed_text(
        model, [chunk.content for chunk in chunks]
    )
    bm25_retriever = BM25Retriever(chunks)

    # 4. Evaluate all three retrieval methods.
    methods = ["Semantic", "BM25", "Hybrid"]
    metrics = ["Recall@3", "Precision@3", "MRR"]
    scores = {
        method: {metric: [] for metric in metrics}
        for method in methods
    }

    for item in evaluation_queries:
        query = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])



        semantic_results = semantic_search(
            query=query,
            chunks=chunks,
            model=model,
            chunk_embedding=chunk_embeddings,
            top_k=3,
            min_similarity=-1.0,
        )

        bm25_results = bm25_retriever.search(query, top_k=3)

        hybrid_results = hybrid_search(
            query=query,
            chunks=chunks,
            model=model,
            chunk_embeddings=chunk_embeddings,
            bm25_retriever=bm25_retriever,
            top_k=3,
            candidate_k=8,
        )

        results_by_method = {
            "Semantic": semantic_results,
            "BM25": bm25_results,
            "Hybrid": hybrid_results,
        }

        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print(f"Relevant chunk IDs: {sorted(relevant_ids)}")

        if not relevant_ids:
            print("\nUNANSWERABLE TEST CASE")
            print("Expected: No supporting chunk in the current document collection.")

            semantic_results = semantic_search(
                query=query,
                chunks=chunks,
                model=model,
                chunk_embedding=chunk_embeddings,
                top_k=3,
                min_similarity=-1.0,
            )

            bm25_results = bm25_retriever.search(query, top_k=3)

            hybrid_results = hybrid_search(
                query=query,
                chunks=chunks,
                model=model,
                chunk_embeddings=chunk_embeddings,
                bm25_retriever=bm25_retriever,
                top_k=3,
                candidate_k=8,
            )

            print("Semantic candidates:", len(semantic_results))
            print("BM25 candidates:", len(bm25_results))
            print("Hybrid candidates:", len(hybrid_results))
            continue

        for method in methods:
            results = results_by_method[method]
            recall = recall_at_k(results, relevant_ids, k=3)
            precision = precision_at_k(results, relevant_ids, k=3)
            mrr = reciprocal_rank(results, relevant_ids)

            scores[method]["Recall@3"].append(recall)
            scores[method]["Precision@3"].append(precision)
            scores[method]["MRR"].append(mrr)

            print(f"\n{method}")
            print(f"  Recall@3:    {recall:.3f}")
            print(f"  Precision@3: {precision:.3f}")
            print(f"  Reciprocal Rank: {mrr:.3f}")

            retrieved_ids = [r["chunk_id"] for r in results[:3]]
            print(f"  Retrieved IDs: {retrieved_ids}")



    # 5. Compare average Recall@3 across all evaluation queries.

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS: MEAN RETRIEVAL METRICS")
    print(f"{'=' * 70}")

    print(f"{'Method':<12} {'Recall@3':>10} {'Precision@3':>13} {'MRR':>10}")

    for method in methods:
        mean_recall = sum(scores[method]["Recall@3"]) / len(evaluation_queries)
        mean_precision = sum(scores[method]["Precision@3"]) / len(evaluation_queries)
        mean_mrr = sum(scores[method]["MRR"]) / len(evaluation_queries)

        print(
            f"{method:<12} "
            f"{mean_recall:>10.3f} "
            f"{mean_precision:>13.3f} "
            f"{mean_mrr:>10.3f}"
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
