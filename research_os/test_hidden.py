# test_hidden.py

from retrieve import HybridRetriever

from generate import (
    generate_hidden_answer,
)

retriever = HybridRetriever()

results = retriever.retrieve(
    "Why does this architecture avoid repeated encoder computation?"
)

ids = [
    r["id"]
    for r in results
]

answer = generate_hidden_answer(
    "Why does this architecture avoid repeated encoder computation?",
    ids,
)

print(answer)