#retrieve.py

from rank_bm25 import BM25Okapi

from models import embedding_model

from qdrant_store import (
    dense_search,
)


class HybridRetriever:

    def __init__(self):

        self.chunks = []

        self.bm25 = None

    def build_bm25(self, chunks):

        self.chunks = chunks

        tokenized = [
            c["text"].split()
            for c in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

    def retrieve(
        self,
        query,
        top_k=2,
    ):

        query_embedding = (
            embedding_model.encode(
                query,
                normalize_embeddings=True,
            )
        )

        dense_results = dense_search(
            query_embedding.tolist(),
            limit=top_k * 2,
        )

        candidates = []

        for r in dense_results:

            candidates.append({
                "id": r.id,
                "text": r.payload["text"],
            })

        self.build_bm25(
            candidates
        )

        bm25_texts = (
            self.bm25.get_top_n(
                query.split(),
                [c["text"] for c in candidates],
                n=top_k,
            )
        )

        final = []

        for text in bm25_texts:

            for candidate in candidates:

                if candidate["text"] == text:

                    final.append(candidate)

                    break

        return final