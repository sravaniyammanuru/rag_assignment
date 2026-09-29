from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL, LLM_MODEL, google_api_key, require_keys


def get_embeddings():
    require_keys()
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key(),
        output_dimensionality=EMBEDDING_DIM,
    )


def get_llm():
    require_keys()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=google_api_key(),
        max_retries=5,
    )
