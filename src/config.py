import os

from dotenv import load_dotenv

load_dotenv(override=True)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
LLM_MODEL = "gemini-3.5-flash-lite"

PDF_PATH = os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6
EMBED_BATCH_SIZE = 8
EMBED_PAUSE_SECONDS = 2.0
WEAK_MATCH_THRESHOLD = 0.35


def _walk_secrets(obj) -> None:
    if obj is None:
        return
    if hasattr(obj, "items"):
        for key, value in obj.items():
            if hasattr(value, "items") and not isinstance(value, (str, bytes)):
                _walk_secrets(value)
            elif value is not None and str(value).strip():
                os.environ[str(key)] = str(value).strip().strip('"').strip("'")
        return
    if isinstance(obj, dict):
        _walk_secrets(obj)


def load_streamlit_secrets() -> None:
    try:
        import streamlit as st

        _walk_secrets(st.secrets)
    except Exception:
        return


def google_api_key() -> str | None:
    load_streamlit_secrets()
    value = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return value.strip() if value else None


def pinecone_api_key() -> str | None:
    load_streamlit_secrets()
    value = os.getenv("PINECONE_API_KEY")
    return value.strip() if value else None


def pinecone_index_name() -> str:
    load_streamlit_secrets()
    return os.getenv("PINECONE_INDEX_NAME", "agentic-ai-gemini")


def pinecone_cloud() -> str:
    load_streamlit_secrets()
    return os.getenv("PINECONE_CLOUD", "aws")


def pinecone_region() -> str:
    load_streamlit_secrets()
    return os.getenv("PINECONE_REGION", "us-east-1")


# kept for scripts that still import the old names
GOOGLE_API_KEY = google_api_key()
PINECONE_API_KEY = pinecone_api_key()
PINECONE_INDEX_NAME = pinecone_index_name()
PINECONE_CLOUD = pinecone_cloud()
PINECONE_REGION = pinecone_region()


def require_keys():
    missing = []
    if not google_api_key():
        missing.append("GOOGLE_API_KEY")
    if not pinecone_api_key():
        missing.append("PINECONE_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing env vars: "
            + ", ".join(missing)
            + ". On Streamlit Cloud, add them under App settings → Secrets (TOML)."
        )
