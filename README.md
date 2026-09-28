# Agentic AI RAG chatbot

Small RAG app over the Agentic AI eBook. Questions go through a LangGraph retrieve -> generate flow. Answers are supposed to stay inside the PDF; if the book doesn't cover it, the model should say so.

I used FastAPI as the main interface and added a Streamlit page because it's nicer for showing chunks on the side.

## Layout

```
rag-agentic-ai/
├── data/Ebook-Agentic-AI.pdf   # source document
├── src/
│   ├── config.py               # env + a few constants
│   ├── models.py               # Gemini embeddings + chat
│   ├── ingestion.py            # PDF -> chunks -> Pinecone
│   ├── retriever.py            # similarity search
│   └── graph.py                # LangGraph state machine
├── app.py                      # FastAPI /chat
├── streamlit_app.py            # optional UI
├── tests_sample_queries.py
├── requirements.txt
└── .env.example
```

Flow is straightforward:

1. `ingestion.py` splits the PDF and upserts embeddings into Pinecone.
2. `/chat` (or Streamlit) sends the question into the graph.
3. `retrieve` pulls the top chunks and a cosine similarity score.
4. `generate` answers from those chunks only.

## Setup

Python 3.10+. On Windows I used a normal venv (no WSL).

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux: `source venv/bin/activate`

Copy `.env.example` to `.env` and put in:

- `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) (free quota)
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default `agentic-ai-gemini`)

If the index doesn't exist yet, ingestion creates a serverless cosine index, dim 768, using `gemini-embedding-001` (`output_dimensionality=768`). Don't point this at an old OpenAI index (those are 1536-d). If create fails, check `PINECONE_CLOUD` / `PINECONE_REGION` — free tier is usually `aws` + `us-east-1`.

## Ingest the PDF

Do this once (uses the Gemini embedding API):

```bash
python -m src.ingestion
```

If you already ingested and just want to rebuild:

```bash
python -m src.ingestion --force
```

`--force` wipes the index first. Running ingest twice without that will duplicate vectors.

If Gemini returns a quota / 429 error, ingest waits and retries. Free-tier embed is capped at 100 requests/min, so the first run can take a couple of minutes.

## Run the API

```bash
python app.py
```

or:

```bash
uvicorn app:app --reload --port 8000
```

Try:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What is Agentic AI according to the eBook?\"}"
```

Response shape:

```json
{
  "answer": "...",
  "retrieved_chunks": [
    {"content": "...", "score": 0.82, "metadata": {"page": 12}}
  ],
  "confidence_score": 0.82
}
```

`confidence_score` is the best Pinecone cosine similarity for that query. If the model refuses, or the match is weak, it gets clamped down so off-topic questions don't look "confident".

## Streamlit (optional)

```bash
streamlit run streamlit_app.py
```

Chat on the left, retrieved chunks + score in the sidebar.

## Sample questions

```bash
python tests_sample_queries.py
```

If the API is already running:

```bash
python tests_sample_queries.py --api
```

Queries baked in:

- What is Agentic AI according to the eBook?
- How do AI agents differ from traditional automation systems?
- What are the core components of an Agentic Architecture?
- What role does memory play in Agentic AI workflows?
- Who won the 2022 FIFA World Cup?  (should refuse)
- What evaluation methods are mentioned for agentic systems?

The World Cup one is the sanity check. Retrieval might still return *some* chunks, but generation should not invent a winner.

## Notes

- LLM is `gemini-3.5-flash-lite`, embeddings are `gemini-embedding-001` truncated to 768-d.
- Graph state: `question`, `context`, `answer`, `score` (plus the raw docs for the API).
- Don't commit `.env`. `.gitignore` already skips it.
