# test_generation.py
from retrieve import HybridRetriever

from generate import (
    generate_text_answer,
)

retriever = HybridRetriever()

results = retriever.retrieve(
    "How does neuron stars form?"
)

chunks = [
    r["text"]
    for r in results
]

answer = generate_text_answer(
    "How does neuron stars form?",
    chunks,
)

print(answer)