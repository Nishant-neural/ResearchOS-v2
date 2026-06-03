# evaluate.py

import json

from retrieve import HybridRetriever

from generate import (
    generate_text_answer,
)


retriever = HybridRetriever()


with open(
    "experiments/evaluation_queries.json"
) as f:

    queries = json.load(f)


results = []

for item in queries:

    query = item["query"]

    chunks = retriever.retrieve(
        query
    )

    answer = generate_text_answer(
        query,
        chunks,
    )

    results.append(
        {
            "query": query,
            "retrieved_chunks": chunks,
            "answer": answer,
        }
    )


with open(
    "experiments/results/results.json",
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )