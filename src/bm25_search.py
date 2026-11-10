from rank_bm25 import BM25Okapi
from src.keyword_search import tokenize

class BM25Retriever:
    def __init__(self, chunks:list):
        self.chunks = chunks

        self.tokenized_chunks = [
            tokenize(chunk.content)
            for chunk in chunks
        ]

        if not chunks or not any (self.tokenized_chunks):
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,

    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.bm25 is None:
            return []

        query_token = tokenize(query)

        if not query_token:
            return[]

        scores = self.bm25.get_scores(query_token)
        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                break

            chunk = self.chunks[int(index)]
            chunk_tokens = set(self.tokenized_chunks[int(index)])

            results.append({
                "score": score,
                "matched_terms": sorted(
                    set(query_token) & chunk_tokens
                ),
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
            }
            )

            if len(results) >= top_k:
                break

        return results