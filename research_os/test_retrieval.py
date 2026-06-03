# test_retrieval.py

from retrieve import HybridRetriever

retriever = HybridRetriever()

results = retriever.retrieve(
    "What problem does RAG solve?"
)

for r in results:
    print()
    print("=" * 80)
    print(r)