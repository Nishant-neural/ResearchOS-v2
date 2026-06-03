# test_hidden.py

from retrieve import HybridRetriever

from generate import (
    generate_hidden_answer,
)

retriever = HybridRetriever()

results = retriever.retrieve(
    "How does neuron stars form?"
)

ids = [
    r["id"]
    for r in results
]

answer = generate_hidden_answer(
    "How does neuron stars form?",
    ids,
)

print(answer)