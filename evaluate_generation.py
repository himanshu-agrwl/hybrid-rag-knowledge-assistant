
import json
from pathlib import Path

from src.document_loader import load_documents
from src.chunker import paragraph_chunks
from src.embeddings import load_embedding_model, embed_text
from src.bm25_search import BM25Retriever
from src.hybrid_search import hybrid_search
from src.generator import RAGGenerator


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENT_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation_generation.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "generation_results.json"


def main():
    # 1. Load evaluation questions
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        evaluation_data = json.load(file)

    # 2. Load and chunk documents
    documents = load_documents(DOCUMENT_DIR)

    chunks = []
    for document in documents:
        chunks.extend(paragraph_chunks(document, max_chars=300))

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.")

    # 3. Initialize retrieval components
    embedding_model = load_embedding_model()

    chunk_embeddings = embed_text(
        embedding_model,
        [chunk.content for chunk in chunks]
    )

    bm25_retriever = BM25Retriever(chunks)

    # 4. Initialize generator
    generator = RAGGenerator()

    results = []

    try:
        # 5. Run every evaluation question
        for test_case in evaluation_data:

            question = test_case["question"]
            expected_behavior = test_case["expected_behavior"]

            print("\n" + "=" * 65)
            print(f"TEST CASE: {test_case['id']}")
            print(f"QUESTION: {question}")
            print(f"EXPECTED BEHAVIOUR: {expected_behavior}")

            # Retrieve relevant context
            retrieved_chunks = hybrid_search(
                query=question,
                chunks=chunks,
                model=embedding_model,
                chunk_embeddings=chunk_embeddings,
                bm25_retriever=bm25_retriever,
                top_k=2,
                candidate_k=8
            )

            # Generate answer
            result = generator.genrate_answer(
                questions=question,
                retrieved_chunks=retrieved_chunks
            )

            print("\nGENERATED ANSWER:")
            print(result["answer"])

            print("\nRETRIEVED SOURCES:")
            for source in result["sources"]:
                print(
                    f"- {source['source']} "
                    f"({source['chunk_id']})"
                )

            # Store results
            results.append({
                "id": test_case["id"],
                "question": question,
                "expected_behavior": expected_behavior,
                "key_points": test_case["key_points"],
                "answer": result["answer"],
                "sources": result["sources"]
            })

    finally:
        generator.close()

    # 6. Save evaluation results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("EVALUATION RUN COMPLETED")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
