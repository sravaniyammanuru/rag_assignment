"""
Hit the compiled graph with the assignment's sample questions.

Run from the project root after ingestion:

    venv\\Scripts\\python tests_sample_queries.py

If the API is already up on :8000 you can pass --api instead.
"""

from __future__ import annotations

import argparse
import sys

SAMPLE_QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]


def run_via_graph(query: str) -> dict:
    from src.graph import rag_graph

    result = rag_graph.invoke(
        {
            "question": query,
            "context": [],
            "documents": [],
            "scores": [],
            "answer": "",
            "score": 0.0,
        }
    )
    return {
        "answer": result["answer"],
        "confidence_score": result["score"],
        "retrieved_chunks": result.get("context") or [],
    }


def run_via_api(query: str, base_url: str) -> dict:
    import httpx

    response = httpx.post(f"{base_url.rstrip('/')}/chat", json={"query": query}, timeout=60.0)
    response.raise_for_status()
    payload = response.json()
    chunks = payload.get("retrieved_chunks") or []
    texts = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            texts.append(chunk.get("content", ""))
        else:
            texts.append(str(chunk))
    if not texts:
        texts = list(payload.get("retrieved_context_chunks") or [])
    return {
        "answer": payload.get("final_answer") or payload.get("answer"),
        "confidence_score": payload.get("confidence_score"),
        "retrieved_chunks": texts or (payload.get("retrieved_context_chunks") or []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="call FastAPI instead of the graph directly")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    runner = (lambda q: run_via_api(q, args.base_url)) if args.api else run_via_graph

    for i, query in enumerate(SAMPLE_QUERIES, start=1):
        print("=" * 72)
        print(f"[{i}/{len(SAMPLE_QUERIES)}] {query}")
        try:
            result = runner(query)
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        print(f"confidence_score: {result['confidence_score']}")
        print("answer:")
        print(result["answer"])
        print(f"retrieved_chunks: {len(result['retrieved_chunks'])}")
        # keep stdout readable; dump full payload with --json if needed
        preview = result["retrieved_chunks"][:1]
        if preview:
            print("first chunk preview:")
            print(preview[0][:400].replace("\n", " "))

    print("=" * 72)
    print("done")


if __name__ == "__main__":
    # so `python tests_sample_queries.py` finds src/
    sys.path.insert(0, ".")
    main()
