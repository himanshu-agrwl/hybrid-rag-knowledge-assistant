
from pathlib import Path

from src.document_loader import load_documents
from src.chunker import paragraph_chunks
from src.embeddings import load_embedding_model, embed_text
from src.bm25_search import BM25Retriever
from src.hybrid_search import hybrid_search
from src.generator import RAGGenerator


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENT_DIR = PROJECT_ROOT / "data" / "documents"


def main() -> None:

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    documents = load_documents(DOCUMENT_DIR)

    print(f"\nLoaded {len(documents)} documents.")

    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    chunks = []

    for document in documents:
        chunks.extend(
            paragraph_chunks(document, max_chars=300)
        )

    if not chunks:
        print("No non-empty chunks found.")
        return

    print(f"Created {len(chunks)} chunks.")

    # --------------------------------------------------
    # 3. Generate embeddings
    # --------------------------------------------------

    model = load_embedding_model()

    chunk_texts = [chunk.content for chunk in chunks]

    chunk_embeddings = embed_text(model, chunk_texts)

    print(f"Embedding matrix shape: {chunk_embeddings.shape}")

    # --------------------------------------------------
    # 4. Initialize retrievers
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(chunks)

    print("BM25 index created.")

    # --------------------------------------------------
    # 5. Initialize LLM generator
    # --------------------------------------------------

    generator = RAGGenerator()


    print("Gemini generator initialized.")

    try:

        # --------------------------------------------------
        # 6. Question-answering loop
        # --------------------------------------------------

        while True:

            query = input(
                "\nAsk a question (or type 'exit'): "
            ).strip()

            if query.lower() == "exit":
                break

            if not query:
                print("Please enter a question.")
                continue

            # --------------------------------------------------
            # 7. Retrieve relevant chunks using hybrid search
            # --------------------------------------------------

            hybrid_results = hybrid_search(
                query=query,
                chunks=chunks,
                model=model,
                chunk_embeddings=chunk_embeddings,
                bm25_retriever=bm25_retriever,
                top_k=2,
            )

            if not hybrid_results:
                print("\nNo relevant chunks were retrieved.")
                print("The available documents may not contain sufficient evidence.")
                continue

            print("\nRetrieved context:")

            for rank, result in enumerate(hybrid_results, start=1):

                print(f"\nRank: {rank}")
                print(f"RRF Score: {result['rrf_score']:.4f}")
                print(f"Retrieved by: {result['retrieved_by']}")
                print(f"Source: {result['source']}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Content: {result['content']}")

            # --------------------------------------------------
            # 8. Generate an answer using Gemini
            # --------------------------------------------------

            print("\nGenerating answer...")

            try:
                answer_result = generator.genrate_answer(
                    questions=query,
                    retrieved_chunks=hybrid_results,
                )

            except RuntimeError as exc:
                print(f"\nGeneration failed: {exc}")
                continue


            # --------------------------------------------------
            # 9. Display answer and sources
            # --------------------------------------------------

            print("\n" + "=" * 60)
            print("GENERATED ANSWER")
            print("=" * 60)

            print(answer_result["answer"])

            print("\nSOURCES")

            for source in answer_result["sources"]:
                print(f"- {source['source']} ({source['chunk_id']})")

            print("=" * 60)

    finally:
        generator.close()
        print("\nGemini client closed.")


if __name__ == "__main__":
    main()
