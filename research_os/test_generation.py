# test_generation.py
from retrieve import HybridRetriever

from generate import (
    generate_text_answer,
)

retriever = HybridRetriever()

results = retriever.retrieve(
    "Why does this architecture avoid repeated encoder computation?"
)

chunks = [
    r["text"]
    for r in results
]

answer = generate_text_answer(
    "Why does this architecture avoid repeated encoder computation?",
    chunks,
)

print(answer)