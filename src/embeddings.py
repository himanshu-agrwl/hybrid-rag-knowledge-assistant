from sentence_transformers import SentenceTransformer
import numpy as np
import torch

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_embedding_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on: {device}")

    return SentenceTransformer(MODEL_NAME, device = device)

def embed_text(
        model: SentenceTransformer,
        text = list[str],

) -> np.ndarray:
    if not text:
        return np.empty((0,384), dtype = np.float32 )

    embeddings = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,

    )

    return embeddings

def semantic_search(
        query: str,
        chunks: list,
        model: SentenceTransformer,
        chunk_embedding: np.ndarray,
        top_k: int =3,
        min_similarity: float = 0.30,
) -> list[dict]:
    if not query.strip():
        raise ValueError("Query must not be empty")

    if top_k <= 0:
        raise ValueError("top k must be greater than 0")

    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1")

    if len(chunks) != len(chunk_embedding):
        raise ValueError("length of chunks and embeddings must match")

    if not chunks:
        return []

    query_embedding = embed_text(model, [query])[0]

    # Embeddings are normalized, so dot product equals cosine similarity

    scores = chunk_embedding @ query_embedding

    rank_indices = np.argsort(scores)[::-1]

    if scores[rank_indices[0]] < min_similarity:
        return []


    results = []
    for index in rank_indices:
        if len(results) >= top_k:
            break

        score = float(scores[index])

        if score < min_similarity:
            break


        chunk = chunks[int(index)]
        results.append(
            {
                "score": float(scores[index]),
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
            }
        )

    return results
