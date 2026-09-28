import os

from dotenv import load_dotenv

load_dotenv(override=True)

# Google AI Studio key (https://aistudio.google.com/apikey)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-gemini")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Gemini embeddings default to 3072-d; we truncate to 768 to keep Pinecone cheap.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
LLM_MODEL = "gemini-3.5-flash-lite"

PDF_PATH = os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6

# Free Gemini embed quota is 100 requests/min; keep batches small and pause.
EMBED_BATCH_SIZE = 8
EMBED_PAUSE_SECONDS = 2.0

# Pinecone cosine scores are similarity (higher = closer).
WEAK_MATCH_THRESHOLD = 0.35


def require_keys():
    missing = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing env vars: " + ", ".join(missing) + ". Copy .env.example to .env and fill them in."
        )
