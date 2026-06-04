# test_hidden.py

from retrieve import HybridRetriever

from generate import (
    generate_hidden_answer,
)

retriever = HybridRetriever()

results = retriever.retrieve(
    "What are galaxies?"
)

ids = [
    r["id"]
    for r in results
]

answer = generate_hidden_answer(
    "What are galaxies?",
    ids,
)

print(answer)