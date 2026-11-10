
from src.document_loader import load_documents
from src.chunker import paragraph_chunks


def main():
    documents = load_documents("data/documents")

    chunks = []
    for document in documents:
        chunks.extend(paragraph_chunks(document, max_chars=300))

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 80)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Chunk index: {chunk.chunk_index}")
        print("-" * 80)
        print(chunk.content)


if __name__ == "__main__":
    main()
