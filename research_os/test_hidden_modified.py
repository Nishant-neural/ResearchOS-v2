# test_hidden.py

from retrieve import HybridRetriever

from generate_modified import (
    generate_hidden_answer,
    inspect_text_encoding,
)

inspect_text_encoding("""
Online Retrieval + Generation user query ↓ retriever ↓ retrieve chunk IDs ↓ load stored hidden states ↓ decoder cross-attention ↓ autoregressive generation At 
inference time: the decoder conditions on latent semantic states, not raw chunk text. The encoder computation disappears entirely during retrieval-time inference. 6. Key Architectural
 Difference Traditional RAG performs: text → semantic understanding for every query. This architecture instead performs: text → semantic understanding once during ingestion. 
 Inference then becomes: query + latent semantic memory ↓ decoder reasoning ↓ generation • • 4 The architecture attempts to reuse: semantic computation itself, not merely retrieved text. 7. Important Tradeoff
  The encoder preprocessing stage is query-independent. Meaning: the encoder does not jointly attend over: user query,
 retrieved chunk, during encoding. This differs from standard RAG where: query + chunk ↓ joint contextual attention occurs dynamically at inference
""")

retriever = HybridRetriever()

results = retriever.retrieve(
    "What are encoder hidden states hypothesized to be?"
)

ids = [
    r["id"]
    for r in results
]

answer = generate_hidden_answer(
    "What are encoder hidden states hypothesized to be?",
    ids,
)

print(answer)
