from langchain_pinecone import PineconeVectorStore

from src.config import PINECONE_INDEX_NAME, TOP_K, require_keys
from src.models import get_embeddings

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        require_keys()
        _vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=get_embeddings(),
        )
    return _vector_store


def retrieve_documents(query: str, k: int = TOP_K):
    return get_vector_store().similarity_search(query, k=k)


def retrieve_documents_with_scores(query: str, k: int = TOP_K):
    return get_vector_store().similarity_search_with_score(query, k=k)
