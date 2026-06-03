# retrieve.py

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
            c.split()
            for c in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

    def retrieve(
        self,
        query,
        top_k=5,
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

        dense_chunks = [
            r.payload["text"]
            for r in dense_results
        ]

        self.build_bm25(
            dense_chunks
        )

        bm25_results = (
            self.bm25.get_top_n(
                query.split(),
                dense_chunks,
                n=top_k,
            )
        )

        return bm25_results