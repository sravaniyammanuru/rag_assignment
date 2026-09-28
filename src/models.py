from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL, GOOGLE_API_KEY, LLM_MODEL, require_keys


def get_embeddings():
    require_keys()
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=EMBEDDING_DIM,
    )


def get_llm():
    require_keys()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        max_retries=5,
    )
