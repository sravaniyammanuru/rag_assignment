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
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["chunk from the PDF..."],
  "confidence_score": 0.82,
  "retrieved_chunks": [
    {"content": "...", "score": 0.82, "metadata": {"page": 12}}
  ]
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

Queries baked in (interview-task set):

- What is the core definition of Agentic AI as outlined in the eBook?
- What are the main architectural components required to build agentic systems?
- What real-world industry use cases for Agentic AI are discussed in the eBook?
- How does Agentic AI differ from traditional generative AI chatbots according to the text?
- What key challenges or limitations of Agentic AI are mentioned in the document?
- What is the capital of France?  (should refuse)

## Notes

- LLM is `gemini-3.5-flash-lite`, embeddings are `gemini-embedding-001` truncated to 768-d.
- Graph state: `question`, `context`, `answer`, `score` (plus the raw docs for the API).
- Don't commit `.env`. `.gitignore` already skips it.

## Publish (Streamlit Cloud)

This is the public demo URL for the status form.

1. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. New app → repo `sravaniyammanuru/rag_assignment` → main → `streamlit_app.py`.
3. App settings → Secrets. Use **TOML** with quotes (not `KEY=value`):

```toml
GOOGLE_API_KEY = "your_gemini_key"
PINECONE_API_KEY = "your_pinecone_key"
PINECONE_INDEX_NAME = "agentic-ai-gemini"
```

4. Deploy. The URL looks like `https://something.streamlit.app`.

Pinecone must already have the eBook vectors (run `python -m src.ingestion` locally once). Cloud only queries the index; it does not re-embed the PDF.

OpenAI API billing is paid. This project uses Gemini (Google AI Studio free quota) for embeddings and chat.
