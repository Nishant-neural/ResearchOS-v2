# test_retrieval.py
from retrieve import HybridRetriever

retriever = HybridRetriever()

results = retriever.retrieve(
    "What is the architecture based on?"
)

print("\nTEXTS:\n")

for r in results:

    print("=" * 80)
    print(r["text"])

print("\nIDS:\n")

for r in results:

    print(r["id"])