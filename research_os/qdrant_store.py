# qdrant_store.py

from qdrant_client import QdrantClient

from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

from config import (
    QDRANT_URL,
    DENSE_COLLECTION,
    HIDDEN_COLLECTION,
)


client = QdrantClient(
    path="./qdrant_data"
)


def create_collections():

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if DENSE_COLLECTION not in collections:

        client.create_collection(
            collection_name=DENSE_COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

    if HIDDEN_COLLECTION not in collections:

        client.create_collection(
            collection_name=HIDDEN_COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )


def upsert_dense(points):

    client.upsert(
        collection_name=DENSE_COLLECTION,
        points=points,
    )


def upsert_hidden(points):

    client.upsert(
        collection_name=HIDDEN_COLLECTION,
        points=points,
    )


def dense_search(
    vector,
    limit=5,
):

    return client.search(
        collection_name=DENSE_COLLECTION,
        query_vector=vector,
        limit=limit,
    )


def hidden_search(
    ids,
):

    return client.retrieve(
        collection_name=HIDDEN_COLLECTION,
        ids=ids,
    )