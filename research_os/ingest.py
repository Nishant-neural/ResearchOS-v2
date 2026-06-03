# ingest.py

import uuid

import torch

from tqdm import tqdm

from qdrant_client.models import (
    PointStruct,
)

from utils import (
    load_pdf,
    clean_text,
    audit_chunk,
)

from chunking import (
    chunk_text,
)

from models import (
    embedding_model,
    tokenizer,
    generation_model,
)

from qdrant_store import (
    create_collections,
    upsert_dense,
    upsert_hidden,
)


@torch.no_grad()
def encode_hidden_states(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    encoder_outputs = (
        generation_model.encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    )

    return {
        "hidden_states": (
            encoder_outputs
            .last_hidden_state
            .cpu()
            .numpy()
        ),
        "attention_mask": (
            inputs.attention_mask
            .cpu()
            .numpy()
        ),
    }


def ingest_pdf(path):

    create_collections()

    raw = load_pdf(path)

    cleaned = clean_text(raw)

    chunks = chunk_text(cleaned)

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
    )

    dense_points = []

    hidden_points = []

    for chunk, embedding in tqdm(
        zip(chunks, embeddings),
        total=len(chunks),
    ):

        audit = audit_chunk(chunk)

        if not audit["valid"]:
            continue

        chunk_id = str(uuid.uuid4())

        hidden = encode_hidden_states(
            chunk
        )

        dense_points.append(
            PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                },
            )
        )

        hidden_points.append(
            PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload={
                    "hidden_states": hidden[
                        "hidden_states"
                    ].tolist(),
                    "attention_mask": hidden[
                        "attention_mask"
                    ].tolist(),
                },
            )
        )

    upsert_dense(dense_points)

    upsert_hidden(hidden_points)

    print("ingestion complete")


if __name__ == "__main__":

    ingest_pdf(
        "research_os/data/pdfs/paper2.pdf"
    )