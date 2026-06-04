# test_hidden.py

from retrieve import HybridRetriever

from generate import (
    generate_hidden_answer,
)

retriever = HybridRetriever()

results = retriever.retrieve(
    "What is the architecture?"
)

ids = [
    r["id"]
    for r in results
]

answer = generate_hidden_answer(
    "What is the architecture?",
    ids,
)

print(answer)