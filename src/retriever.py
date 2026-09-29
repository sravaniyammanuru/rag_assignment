from langchain_pinecone import PineconeVectorStore

from src.config import pinecone_index_name, require_keys, TOP_K
from src.models import get_embeddings

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        require_keys()
        _vector_store = PineconeVectorStore(
            index_name=pinecone_index_name(),
            embedding=get_embeddings(),
        )
    return _vector_store


def retrieve_documents(query: str, k: int = TOP_K):
    return get_vector_store().similarity_search(query, k=k)


def retrieve_documents_with_scores(query: str, k: int = TOP_K):
    return get_vector_store().similarity_search_with_score(query, k=k)
